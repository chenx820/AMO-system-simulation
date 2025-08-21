import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import qutip as qt

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

# --- Part 1: Atom Array Generation ---
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

# --- Part 2: QuTiP Operator and Hamiltonian Construction ---

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

# --- Part 3: QuTiP-based Simulation Execution ---

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
    pbar = tqdm(pulse_sequence, desc="Simulating Pulses")
    for direction, duration, steps in pbar:
        pbar.set_description(f"Pulse: {direction}")
        
        current_delta = params['V_map'][direction] + params['delta_detuning_map'][direction]
        # Pre-build Hamiltonian for this pulse to avoid repeated construction
        H_pulse = build_hamiltonian(Nx, Ny, params['Omega'], current_delta, params['V_map'])
        H_func = lambda t, args: H_pulse
        
        t_pulse = np.linspace(t_total, t_total + duration, steps + 1)
        # Optimize solver options for 4x4 system
        options = {"store_final_state": True, "nsteps": 1000}
        
        if params['mode'] == 'Schrodinger':
            result = qt.sesolve(H_func, psi0, t_pulse, e_ops=e_ops, options=options)
        elif params['mode'] == 'Lindblad': # Lindblad
            # For mesolve, the initial state can be a state vector or density matrix
            rho0 = psi0 * psi0.dag() if psi0.isket else psi0
            c_ops = get_collapse_operators(N, params['Gamma_decay'], params['Gamma_dephasing'])
            result = qt.mesolve(H_func, rho0, t_pulse, c_ops=c_ops, e_ops=e_ops, options=options)
        else:
            raise ValueError(f"Invalid simulation mode: {params['mode']}")
        
        # Append results, excluding the first point which is the end of the last pulse
        time_points.extend(result.times[1:])
        # result.expect is a list of arrays, one for each e_op. We need to transpose it.
        pulse_history = np.array(result.expect).T
        history.extend(pulse_history[1:])

        # The final state of this pulse is the initial state for the next
        psi0 = result.final_state
        t_total += duration

    return time_points, history

# --- Part 4: Analysis and Visualization ---
def plot_pulse_sequence(ax, pulse_sequence, params, t0=0.0, time_unit_scale=1e6, *, normalize_by_total=False, total_duration=None):
    """
    Plot the pulse sequence on the given axis.
    """
    ts = []
    ys = []
    t = t0
    for direction, duration, steps in pulse_sequence:
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

    # total duration of the pulse sequence
    total_duration = sum(duration for _, duration, _ in pulse_sequence) if len(pulse_sequence) > 0 else None

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
    ax_top.set_xlim(0, time_x[-1] if len(time_x) > 0 else 0)

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
    cbar.ax.set_ylabel('Rydberg population', fontsize=16)  # Colorbar label fontsize

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


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. Simulation Setup ---
    # For larger systems (4x4 or bigger), use 'Schrodinger' mode to avoid memory issues
    # 'Lindblad' mode requires too much memory for systems with >12 atoms
    SIMULATION_MODE = 'Schrodinger'  # 'Schrodinger' or 'Lindblad'
    Nx, Ny = 4, 4
    
    # Check system size and warn about memory requirements
    total_atoms = Nx * Ny
    hilbert_dim = 2**total_atoms
    if SIMULATION_MODE == 'Lindblad' and total_atoms > 12:
        print(f"WARNING: {total_atoms} atoms = {hilbert_dim} Hilbert dimension")
        print(f"Lindblad mode requires ~{hilbert_dim**2 * 16 / 1e9:.1f} GB memory")
        print("Switching to Schrodinger mode to avoid memory overflow...")
        SIMULATION_MODE = 'Schrodinger' 
    
    # --- 2. Physical Parameters (all in SI units) ---
    # Spacings in meters (m)
    r_h1, r_h2 = 11.4e-6, 12.8e-6
    r_v1, r_v2 = 10.8e-6, 14.2e-6

    # Angular frequencies in rad/s
    Omega = 2 * np.pi * 3.0e6  # 3.0 MHz
    Omega_eff = Omega / np.sqrt(2) # effective Rabi frequency
    Gamma_decay = 0.002 * Omega # decay rate
    Gamma_dephasing = 0.004 * Omega # dephasing rate
    
    # van der Waals coefficient in J * m^6. We use angular frequency units (rad/s * m^6)
    # by implicitly setting hbar=1.
    C6 = 3e7 * Omega * (1e-6)**6 # Convert to (rad/s) * m^6
    
    # --- 3. Derived Parameters and Array Generation ---
    atom_coords = create_alternating_2d_array(Nx, Ny, r_h1, r_h2, r_v1, r_v2)
    V_map = {
        'h1': C6 / r_h1**6, 'h2': C6 / r_h2**6,
        'v1': C6 / r_v1**6, 'v2': C6 / r_v2**6
    }
    delta_detuning_map = {
        'h1': -0.133 * Omega, 'h2': -0.033 * Omega,
        'v1': -0.166 * Omega, 'v2': -0.016 * Omega
    }
    
    pi_pulse_duration = np.pi / Omega_eff # Duration in seconds
    
    params = {
        'mode': SIMULATION_MODE, 
        'Omega': Omega, 
        'V_map': V_map, 
        'Gamma_decay': Gamma_decay, 
        'Gamma_dephasing': Gamma_dephasing, 
        'delta_detuning_map': delta_detuning_map
    }

    print("--- Simulation Parameters ---")
    print(f"Grid: {Nx}x{Ny}, Mode: {SIMULATION_MODE}")
    print(f"Omega: {Omega/(2*np.pi*1e6):.2f} MHz")
    
    # Corrected print statement logic
    spacing_values = {'h1': r_h1, 'h2': r_h2, 'v1': r_v1, 'v2': r_v2}
    for key, val in V_map.items():
        r_val = spacing_values[key]
        print(f"V_{key} (r={r_val*1e6:.1f} um): {val/(2*np.pi*1e6):.2f} MHz")

    print(f"Estimated pi-pulse duration: {pi_pulse_duration*1e6:.2f} us")
    if SIMULATION_MODE == 'Lindblad':
        print(f"Gamma_decay: {Gamma_decay/(2*np.pi*1e3):.3f} kHz, Gamma_dephasing: {Gamma_dephasing/(2*np.pi*1e3):.3f} kHz")

    # --- 4. Define Transport Path and Run Simulation ---
    initial_sites = [0]
    target_sites = [10]  
    intermediate_sites = [1, 5, 6]
    
    steps_per_pulse = 50    
    pulse_sequence = [
        ('h1', pi_pulse_duration, steps_per_pulse),  
        ('v1', pi_pulse_duration, steps_per_pulse),
        ('h2', pi_pulse_duration, steps_per_pulse),  
        ('v2', pi_pulse_duration, steps_per_pulse),  
    ] 
    
    times, history = run_simulation(Nx, Ny, params, pulse_sequence, initial_sites)

    # --- 5. Analyze and Visualize Results ---
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
        # plot the transport dynamics and the pulse sequence
        plot_transport_and_pulses(times, history, pulse_sequence, params, sites_for_plot)
        animate_transport(Nx, Ny, atom_coords, times, history)

        # the population snapshot at the middle and final time
        mid_idx = len(history) // 2
        plot_population_snapshot(history[mid_idx], Nx, Ny, t=times[mid_idx], filename='transport_2d_snapshot_mid.png')
        plot_population_snapshot(history[-1], Nx, Ny, t=times[-1], filename='transport_2d_snapshot_final.png')
        print("Saved population snapshots: 'transport_2d_snapshot_mid.png', 'transport_2d_snapshot_final.png'")
    else:
        print("\nSimulation did not produce data. Check parameters.")