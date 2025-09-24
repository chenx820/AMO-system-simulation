#!/usr/bin/env python3
"""
2D Transport Simulation Script
simulate the transport of Rydberg atoms in a 2D grid

Author: Chen Huang
Date: 2025-08-22
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import qutip as qt
import json
import os

from src.system_params import create_custom_system

def load_config(config_path='./configs/default_params.json', example_path=None):
    """
    load parameters from JSON config file and optionally merge with example config
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Warning: config file {config_path} not found, using default parameters")
    
    # if example_path is provided, load and merge with config
    if example_path:
        try:
            with open(example_path, 'r') as f:
                example_config = json.load(f)
            
            # merge example config with main config
            if 'initial_sites' in example_config:
                config['initial_sites'] = example_config['initial_sites']
            if 'target_sites' in example_config:
                config['target_sites'] = example_config['target_sites']
            if 'pulse_sequence_config' in example_config:
                config['pulse_sequence_config'] = example_config['pulse_sequence_config']
            if 'info_path' in example_config:
                config['info_path'] = example_config['info_path']
            
            print(f"Loaded example config from {example_path}")
            print(f"Initial sites: {config.get('initial_sites', 'not set')}")
            print(f"Target sites: {config.get('target_sites', 'not set')}")
            print(f"Pulse sequence: {config.get('pulse_sequence_config', 'not set')}")
            print(f"Info path: {config.get('info_path', 'not set')}")
            
        except FileNotFoundError:
            print(f"Warning: example config file {example_path} not found")
        except json.JSONDecodeError:
            print(f"Warning: example config file {example_path} JSON format error")
    
    return config

# define the single-atom operators
sm = qt.destroy(2) # lowering operator |0><1|
sp = qt.create(2)  # raising operator |1><0|
sz = qt.sigmaz()   # Pauli Z operator |0><0| - |1><1|
n_op = sp * sm  # number operator |1><1|

# Plot settings
plt.style.use('seaborn-v0_8-paper')
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = ['Times New Roman']

colors = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0C241",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#A6A2A2",
    "#01766E",
    "#A17724",
    "#581845",
]

# --- Atom Array Generation ---
def create_alternating_2d_array(Nx, Ny, r_h1, r_h2, r_v1, r_v2):
    """
    Creates coordinates for an n x n 2D atom array with alternating spacings.
    This geometry is crucial for creating distinct interaction energies for directional transport.
    """
    coordinates = []
    y_positions = [0.0] * Ny
    x_positions = [0.0] * Nx
    for i in range(1, Ny):
        y_positions[i] = y_positions[i-1] + (r_v1 if (i - 1) % 2 == 0 else r_v2)
    for j in range(1, Nx):
        x_positions[j] = x_positions[j-1] + (r_h1 if (j - 1) % 2 == 0 else r_h2)
    
    for i in range(Ny):
        for j in range(Nx):
            coordinates.append((x_positions[j], y_positions[i]))
            
    return np.array(coordinates)

# --- QuTiP Operator and Hamiltonian Construction ---
def get_operator_for_site(op, i, N):
    """
    Creates a full Hilbert space operator for a single-qubit operator 'op' acting on site 'i',
    using QuTiP's tensor product functionality.
    """
    op_list = [qt.qeye(2)] * N
    op_list[i] = op

    return qt.tensor(op_list)
    

def build_hamiltonian(Nx, Ny, Omega, current_delta, V_map):
    """
    Builds the system Hamiltonian using QuTiP objects.
    H = sum_j (Omega/2 * sigma_x_j - current_delta * n_j) + sum_{i<j} V_ij * n_i * n_j
    """
    H = 0
    N = Nx * Ny
    
    # Driving term
    for i in range(N):
        H += 0.5 * Omega * get_operator_for_site(qt.sigmax(), i, N)
        H -= current_delta * get_operator_for_site(n_op, i, N)

    # Interaction term for nearest neighbors
    for i in range(N):
        row_i, col_i = divmod(i, Nx)
        for j in range(i + 1, N):
            row_j, col_j = divmod(j, Nx)

            V_ij = 0
            # Horizontal neighbors
            if row_i == row_j and abs(col_i - col_j) == 1:
                V_ij = V_map['h1'] if min(col_i, col_j) % 2 == 0 else V_map['h2']
            # Vertical neighbors
            elif col_i == col_j and abs(row_i - row_j) == 1:
                V_ij = V_map['v1'] if min(row_i, row_j) % 2 == 0 else V_map['v2']
            
            if V_ij > 0:
                n_i_n_j = get_operator_for_site(n_op, i, N) * get_operator_for_site(n_op, j, N)
                H += V_ij * n_i_n_j
                
    return H


# Define the collapse operators (Lindblad master equation)
def get_collapse_operators(N, Gamma_decay_val, Gamma_dephasing_val):
    """
    Build the collapse operators in the Lindblad master equation
    """
    c_ops = []
    # Decay term L_decay(rho)
    for k in range(N):
        c_ops.append(np.sqrt(Gamma_decay_val) * get_operator_for_site(sm, k, N))

    # Dephasing term L_deph(rho)
    for k in range(N):
        c_ops.append(np.sqrt(Gamma_dephasing_val) * get_operator_for_site(sz, k, N))
    return c_ops

# --- QuTiP-based Simulation Execution ---
def run_simulation(Nx, Ny, params, pulse_sequence, initial_sites=[0]):
    """
    Runs the time-evolution using QuTiP's sesolve (Schrödinger) or mesolve (Lindblad).
    """
    N = Nx * Ny
    
    # --- Initial State ---
    # |g...1...g> -> tensor product of basis states
    initial_state_list = [qt.basis(2, 0)] * N
    for initial_site in initial_sites:
        initial_state_list[initial_site] = qt.basis(2, 1)
    psi0 = qt.tensor(initial_state_list)

    # --- Expectation operators to track Rydberg population ---
    e_ops = [get_operator_for_site(n_op, i, N) for i in range(N)]

    # --- Evolve the system pulse by pulse ---
    history = []
    time_points = []
    t_total = 0.0
    
    # Use tqdm for progress bar
    print(pulse_sequence)
    pbar = tqdm(pulse_sequence, desc="Simulating Pulses")
    for i, pulse in enumerate(pbar):
        # support two forms:
        # ['h1', duration, steps]
        # ['h1', duration, steps, delta_override]
        if len(pulse) == 3:
            direction, duration, steps = pulse
            delta_override = None
        else:
            direction, duration, steps, delta_override = pulse
        pbar.set_description(f"Pulse: {direction}")
        
        if delta_override is not None:
            current_delta = params['V_map'][direction] + delta_override
        else:
            current_delta = params['V_map'][direction] + params['delta_detuning_map'][direction]
        # Pre-build Hamiltonian for this pulse to avoid repeated construction
        H_pulse = build_hamiltonian(Nx, Ny, params['Omega'], current_delta, params['V_map'])
        H_func = lambda t, args: H_pulse
        
        t_pulse = np.linspace(t_total, t_total + duration, steps + 1)
        # Use optimized solver options
        temp_system = create_custom_system(Nx, Ny)
        options = temp_system.get_optimized_solver_options(duration)
        
        if params['mode'] == 'Schrodinger':
            result = qt.sesolve(H_func, psi0, t_pulse, e_ops=e_ops, options=options)
        elif params['mode'] == 'Lindblad': # Lindblad
            # For mesolve, the initial state can be a state vector or density matrix
            rho0 = psi0 * psi0.dag() if psi0.isket else psi0
            c_ops = get_collapse_operators(N, params['Gamma_decay'], params['Gamma_dephasing'])
            result = qt.mesolve(H_func, rho0, t_pulse, c_ops=c_ops, e_ops=e_ops, options=options)
        else:
            raise ValueError(f"Invalid simulation mode: {params['mode']}")
        
        # For the first pulse, include the initial point; for subsequent pulses, exclude the first point to avoid duplication
        if i == 0:
            time_points.extend(result.times)
            pulse_history = np.array(result.expect).T
            history.extend(pulse_history)
        else:
            time_points.extend(result.times[1:])
            pulse_history = np.array(result.expect).T
            history.extend(pulse_history[1:])

        # The final state of this pulse is the initial state for the next
        psi0 = result.final_state
        t_total += duration

    return time_points, history

# --- Analysis and Visualization ---
def plot_pulse_sequence(ax, pulse_sequence, params, t0=0.0, time_unit_scale=1e6, *, normalize_by_total=False, total_duration=None):
    """
    Plot the pulse sequence on the given axis.
    """
    ts = []
    ys = []
    t = t0
    for pulse in pulse_sequence:
        if len(pulse) == 3:
            direction, duration, steps = pulse
            delta_override = None
        else:
            direction, duration, steps, delta_override = pulse
        if delta_override is not None:
            delta = params['V_map'][direction] + delta_override
        else:
            delta = params['V_map'][direction] + params['delta_detuning_map'][direction]
        # step: stay at current value from t to t+duration
        ts.extend([t, t + duration])
        ys.extend([delta, delta])
        t += duration

    ts = np.array(ts)
    if normalize_by_total and (total_duration is not None) and (total_duration > 0):
        ts = ts / (total_duration)
    elif time_unit_scale is not None:
        ts = ts * time_unit_scale
    ys = np.array(ys) 

    ax.plot(ts, ys, linewidth=4, color=colors[0], drawstyle='steps-post', label='Laser detuning pulse sequence')

    ax.set_xlim(ts[0], ts[-1])
    ax.set_yticks(list(params['V_map'].values()))
    ax.set_yticklabels(list(r'$V_{'+key+'}$' for key in params['V_map'].keys()))

    if normalize_by_total:
        ax.set_xlabel(r'$\tilde{\Omega}t/(2\pi)$', fontsize=20)
    else:
        ax.set_xlabel(r'Time ($\mu$s)', fontsize=20)

    ax.grid(axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylabel(r'$\Delta$', fontsize=20)
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=20)

def plot_transport_and_pulses(time_points, history, pulse_sequence, params, sites_to_plot,
                              filename='transport_2d_dynamics_with_pulses.png'):
    """
    Plot the transport dynamics and the pulse sequence on the given axes.
    """
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, squeeze=False)

    # total duration of the pulse sequence (supports 3-tuple and 4-tuple pulses)
    total_duration = sum((p[1] for p in pulse_sequence)) if len(pulse_sequence) > 0 else None

    # top: dynamics (x-axis: t/T)
    ax_top = axes[0][0]
    hist = np.array(history)
    if total_duration and total_duration > 0:
        time_x = np.array(time_points) / (total_duration)
    else:
        time_x = np.array(time_points) * 1e6  # fallback: if no T, still use μs
    for i, (site_idx, label, style) in enumerate(sites_to_plot):
        ax_top.plot(time_x, hist[:, site_idx], color=colors[i % len(colors)], linestyle=style, linewidth=4, label=label)

    ax_top.set_ylabel('Rydberg population', fontsize=20)
    ax_top.legend(loc='best', frameon=True, ncol=1, framealpha=0.9, fontsize=20)
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.8)
    ax_top.set_xlim(time_x[0] if len(time_x) > 0 else 0, time_x[-1] if len(time_x) > 0 else 0)

    # bottom: pulse sequence (axes[1][0])
    ax_bottom = axes[1][0]
    plot_pulse_sequence(
        ax_bottom, pulse_sequence, params,
        t0=0.0,
        time_unit_scale=1e6,
        normalize_by_total=bool(total_duration and total_duration > 0),
        total_duration=total_duration,
    )

    for ax in axes.flat:
        ax.tick_params(axis='y', direction='in', width=2, length=4 , pad=6 , right=True, labelsize=18)
        ax.tick_params(axis='x', direction='in', width=2, length=6 , pad=6 , top=False, labelsize=18)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(filename, dpi=800)
    print(f"Plot saved to '{filename}'")
    plt.close()

def plot_population_snapshot(pop_vec, Nx, Ny, t=None, filename='population_snapshot.png'):
    """
    The 1D population vector of length N=Nx*Ny is reshaped into a 2D grid of size Nx*Ny.
    Each cell displays the corresponding atom index and population value.
    """
    grid = np.zeros((Ny, Nx))
    for row in range(Ny):
        for col in range(Nx):
            idx = row * Nx + col
            grid[row, col] = pop_vec[idx]

    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin='lower', cmap='cividis', vmin=0, vmax=1)

    for row in range(Ny):
        for col in range(Nx):
            idx = row * Nx + col
            ax.text(col, row, f'{idx}\n{grid[row,col]:.2f}',
                    ha='center', va='center', color='white', fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(direction='out', labelsize=14)  # Colorbar ticks
    cbar.ax.set_ylabel('Rydberg population', fontsize=16)  # Colorbar label

    ax.set_xticks(np.arange(0, Nx, 1))
    ax.set_yticks(np.arange(0, Ny, 1))
    ax.set_xlabel(r'Column index $i$', fontsize=20)
    ax.set_ylabel(r'Row index $j$', fontsize=20)

    ax.tick_params(axis='y', direction='out', width=1.5, length=4, pad=6, right=False, labelsize=18)
    ax.tick_params(axis='x', direction='out', width=1.5, length=4, pad=6, top=False, labelsize=18)
        
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=800)
    plt.close()

def animate_transport(Nx, Ny, coords, time_points, history, filename='transport_2d_animation.mp4'):
    """
    Creates and saves a MP4 video of the transport with site labels.
    NOTE: This requires the 'ffmpeg' package to be installed on your system.
          You can install it via conda: `conda install -c conda-forge ffmpeg`
          or check the ffmpeg website for other installation options.
    """
    fig, ax = plt.subplots(figsize=(Nx+1, Ny+1))
    
    history_norm = np.array(history)
    vmax = np.max(history_norm) if np.max(history_norm) > 0 else 1.0

    # Convert coordinates to micrometers for display
    coords_um = coords * 1e6
    
    # Calculate Ny from coordinates
    Ny = len(coords) // Nx

    def update(frame):
        ax.clear()
        populations = history_norm[frame]
        colors = populations
        sizes = 2000 * populations + 50
        
        ax.scatter(coords_um[:, 0], coords_um[:, 1], c=colors, s=sizes, cmap='viridis', vmin=0, vmax=vmax, zorder=2)
        
        # Add site labels with coordinates and site number
        for i in range(len(coords)):
            row, col = divmod(i, Nx)
            site_label = f"({col},{row})\n{i}"
            ax.annotate(site_label, 
                       (coords_um[i, 0], coords_um[i, 1]), 
                       xytext=(10, 5), 
                       textcoords='offset points',
                       ha='center', va='center',
                       fontsize=8,
                       color='black')
        
        # Draw grid lines
        for i in range(Ny):
            ax.axhline(coords_um[i*Nx, 1], color='grey', linestyle='--', linewidth=0.5, zorder=1)
        for j in range(Nx):
            ax.axvline(coords_um[j, 0], color='grey', linestyle='--', linewidth=0.5, zorder=1)

        ax.set_title(f'2D Excitation Transport\nTime: {time_points[frame]*1e6:.2f} μs')
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_aspect('equal')
        ax.set_xlim(coords_um[:, 0].min() - 2, coords_um[:, 0].max() + 2)
        ax.set_ylim(coords_um[:, 1].min() - 2, coords_um[:, 1].max() + 2)

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50)
    # Use the 'ffmpeg' writer for MP4 output. Set a high DPI for quality and fixed FPS.
    ani.save(filename, writer='ffmpeg', dpi=800, fps=20)
    print(f"Animation saved to '{filename}'")
    plt.close()


# --- Run simulation ---
def simulate_transport(Nx: int, Ny: int, Omega_MHz: float = 3.0, *,
                       detuning_override: dict | None = None,
                       initial_sites: list[int] | None = None,
                       target_sites: list[int] | None = None,
                       intermediate_sites: list[int] | None = None,
                       plots_path: str = None,
                       anim_path: str = None,
                       config: dict = None,
                       pulse_sequence_config: list[tuple[str, float, int]] = None,
                       ):

    # Defaults
    if initial_sites is None:
        initial_sites = [0]
    if target_sites is None:
        target_sites = [1]
    if intermediate_sites is None:
        intermediate_sites = []

    # helper to append suffix before file extension
    def add_suffix_before_ext(path: str, suffix: str) -> str:
        base, ext = os.path.splitext(path)
        return f"{base}{suffix}{ext}"

    # use output paths from config (if provided)
    if config and 'output_paths' in config:
        output_paths = config['output_paths']
        if plots_path is None:
            plots_path = f"./{output_paths['plots_dir']}/transport_2d_dynamics_with_pulses.png"
        if anim_path is None:
            anim_path = f"./{output_paths['animations_dir']}/transport_2d_animation.mp4"
    else:
        # default paths
        if plots_path is None:
            plots_path = './outputs/plots/transport_2d_dynamics_with_pulses.png'
        if anim_path is None:
            anim_path = './outputs/animations/transport_2d_animation.mp4'

    # append size suffix to filenames
    size_suffix = f"_{Nx}x{Ny}"
    plots_path = add_suffix_before_ext(plots_path, size_suffix)
    anim_path = add_suffix_before_ext(anim_path, size_suffix)

    # 1) System parameters
    system_params = create_custom_system(Nx, Ny, Omega_MHz=Omega_MHz)
    if detuning_override:
        # override provided keys only (values should be absolute in rad/s)
        for k, v in detuning_override.items():
            if k in system_params.delta_detuning_map:
                system_params.delta_detuning_map[k] = v

    system_params.print_system_info()

    # 2) Atom coordinates
    atom_coords = create_alternating_2d_array(
        system_params.Nx, system_params.Ny,
        system_params.r_h1, system_params.r_h2,
        system_params.r_v1, system_params.r_v2
    )

    # 3) Pulse sequence
    if pulse_sequence_config and len(pulse_sequence_config) > 0:
        first_item = pulse_sequence_config[0]
        # If already a concrete pulse list like [('h1', duration, steps) or (..., delta_override)] use directly
        if isinstance(first_item, (list, tuple)) and len(first_item) in (3, 4) and isinstance(first_item[0], str):
            pulse_sequence = pulse_sequence_config
        else:
            # Otherwise, treat as high-level spec and build using system params
            pulse_sequence = system_params.create_pulse_sequence(pulse_sequence_config)
    else:
        raise ValueError("Pulse sequence config is not set")

    # 4) Run simulation
    times, history = run_simulation(
        system_params.Nx, system_params.Ny,
        system_params.get_params_dict(),
        pulse_sequence,
        initial_sites
    )

    # 5) Results and outputs
    if history:
        final_populations = history[-1]
        print(f"\n--- Results ---")
        for target_site in target_sites:
            fidelity = final_populations[target_site]
            print(f"Transport Fidelity to site {target_site}: {fidelity:.4f}")

        sites_for_plot = [
            (initial_site, f'Initial site ({initial_site})', '-') for initial_site in initial_sites] + [
            (intermediate_site, f'Intermediate site ({intermediate_site})', '--') for intermediate_site in intermediate_sites] + [
            (target_site, f'Target site ({target_site})', '-') for target_site in target_sites
        ]

        plot_transport_and_pulses(times, history, pulse_sequence, 
                                  system_params.get_params_dict(), sites_for_plot,
                                  filename=plots_path)
        animate_transport(system_params.Nx, system_params.Ny, atom_coords, times, history,
                          filename=anim_path)

        # snapshots
        if config and 'output_paths' in config:
            plots_dir = f"./{config['output_paths']['plots_dir']}"
        else:
            plots_dir = './outputs/plots'
            

        final_path = add_suffix_before_ext(f'{plots_dir}/transport_2d_snapshot_final.png', size_suffix)

        plot_population_snapshot(history[-1], system_params.Nx, system_params.Ny, t=times[-1],
                                filename=final_path)
        print(f"Saved population snapshots: '{final_path}'")
    else:
        print("\nSimulation did not produce data. Check parameters.")


def run_tranport_2d(path):
    from src.optimize_detuning import optimize_path_stepwise

    # load config
    if path:
        config = load_config(example_path=path)
    else:
        raise ValueError("Path to example JSON is not set")
    
    # load system parameters from config
    Nx = config['system']['Nx']
    Ny = config['system']['Ny']
    Omega_MHz_default = config['system']['Omega_MHz'] # MHz
    Omega = 2 * np.pi * Omega_MHz_default * 1e6

    # get initial sites, target sites, and pulse sequence from config
    initial_sites = config.get('initial_sites', [0])
    target_sites = config.get('target_sites', [1])
    intermediate_sites = config.get('info_path', [])
    pulse_sequence_config = config.get('pulse_sequence_config', [])
    optimization = config.get('optimization', False)

    # load optimization parameters
    opt_config = config['optimization']
    
    system_params = create_custom_system(Nx, Ny, Omega_MHz=Omega_MHz_default)

    if optimization:
        print("\nRunning detuning optimization before transport...")

        delta_range = opt_config['delta_range']
        coarse_steps = opt_config['coarse_steps']
        fine_window = opt_config['fine_window']
        fine_steps = opt_config['fine_steps']

        # direction sequence
        directions = []
        for item in pulse_sequence_config:
            # item can be ['h1', 1, 25] or ('h1', 1, 25)
            directions.append(item[0] if isinstance(item, (list, tuple)) else item)

        # target sequence: all intermediate points are maximized sequentially, the last one is the final target
        targets_chain = list(intermediate_sites) + [target_sites[0]]
        # normalize targets_chain length to match directions
        if len(targets_chain) == 0:
            targets_chain = [target_sites[0]] * len(directions)
        elif len(targets_chain) < len(directions):
            targets_chain = targets_chain + [targets_chain[-1]] * (len(directions) - len(targets_chain))
        elif len(targets_chain) > len(directions):
            targets_chain = targets_chain[:len(directions)]

        # steps: (direction, target_site)
        steps = list(zip(directions, targets_chain))

        pulses = optimize_path_stepwise(
            Nx, Ny, Omega,
            steps=steps, 
            start_site=initial_sites[0],
            periods=opt_config.get('periods', 1),
            steps_per_pulse=opt_config.get('steps_per_pulse', 25),
            delta_range=delta_range, coarse_steps=coarse_steps,
            fine_window=fine_window, fine_steps=fine_steps,
        )

        # get initial sites from config
        initial_sites = config.get('initial_sites', [])
        if len(initial_sites) == 0:
            raise ValueError("Initial sites are not set")
        
        times, history = run_simulation(
            system_params.Nx, system_params.Ny,
            system_params.get_params_dict(),
            pulses, initial_sites
        )
        if history:
            simulate_transport(Nx, Ny, Omega_MHz=Omega_MHz_default,
                               detuning_override=None,
                               initial_sites=initial_sites,
                               target_sites=target_sites,
                               intermediate_sites=intermediate_sites,
                               config=config,
                               pulse_sequence_config=pulses)
        else:
            print("Stepwise optimized pulses produced no data; falling back to alternating result.")
            detuning_override = {k: v * Omega for k, v in config['detuning'].items()}
            simulate_transport(
                Nx, Ny, Omega_MHz=Omega_MHz_default, 
                detuning_override=detuning_override, 
                initial_sites=initial_sites,
                target_sites=target_sites,
                intermediate_sites=intermediate_sites,
                config=config,
                pulse_sequence_config=pulse_sequence_config
                )

    else:
        detuning_override = {}
        for k, v in config['detuning'].items():
            detuning_override[k] = v * Omega

        print(
            "Applying detuning from config: \n"
            f"δ_h1 = {detuning_override['h1']/Omega:+.3f}Ω \n"
            f"δ_h2 = {detuning_override['h2']/Omega:+.3f}Ω \n"
            f"δ_v1 = {detuning_override['v1']/Omega:+.3f}Ω \n"
            f"δ_v2 = {detuning_override['v2']/Omega:+.3f}Ω \n"
        )

        if len(pulse_sequence_config) > 0:
            for pulse in pulse_sequence_config:
                pulse[1] = pulse[1] * system_params.pi_pulse_duration
        
        simulate_transport(
            Nx, Ny, Omega_MHz=Omega_MHz_default,
            detuning_override=detuning_override,
            initial_sites=initial_sites,
            target_sites=target_sites,
            intermediate_sites=intermediate_sites,
            config=config,
            pulse_sequence_config=pulse_sequence_config
            )