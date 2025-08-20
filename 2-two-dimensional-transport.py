import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import qutip as qt

# define the single-atom operators
sm = qt.destroy(2) # lowering operator |0><1|
sp = qt.create(2)  # raising operator |1><0|
sz = qt.sigmaz()   # Pauli Z operator |1><1| - |0><0|
n_op = sp * sm  # number operator |1><1|

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
    H = sum_j (Omega/2 * sigma_x_j - current_delta * sigma_z_j) + sum_{i<j} V_ij * n_i * n_j
    """
    H = 0
    N = Nx * Ny
    
    # Driving term
    for i in range(N):
        H += 0.5 * Omega * get_operator_for_site(qt.sigmax(), i, N)
        H -= current_delta * get_operator_for_site(n_op, i, N)

    # Interaction term for nearest neighbors
    for i in range(N):
        for j in range(i + 1, N):
            row_i, col_i = divmod(i, Nx)
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

def run_simulation(Nx, Ny, params, pulse_sequence, initial_site=0):
    """
    Runs the time-evolution using QuTiP's sesolve (Schrödinger) or mesolve (Lindblad).
    """
    N = Nx * Ny
    
    # --- Initial State ---
    # |g...1...g> -> tensor product of basis states
    initial_state_list = [qt.basis(2, 0)] * N
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
        H_func = lambda t, args: build_hamiltonian(
            Nx, Ny, params['Omega'], current_delta, params['V_map']
            )
        
        t_pulse = np.linspace(t_total, t_total + duration, steps + 1)
        
        options = {"store_final_state": True, "nsteps": 5000}  
        
        if params['mode'] == 'Schrodinger':
            result = qt.sesolve(H_func, psi0, t_pulse, e_ops=e_ops, options=options)
        else: # Lindblad
            # For mesolve, the initial state can be a state vector or density matrix
            rho0 = psi0 * psi0.dag() if psi0.isket else psi0
            c_ops = get_collapse_operators(N, params['Gamma_decay'], params['Gamma_dephasing'])
            result = qt.mesolve(H_func, rho0, t_pulse, c_ops=c_ops, e_ops=e_ops, options=options)
        
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

def plot_transport_dynamics(time_points, history, sites_to_plot, filename='transport_dynamics.png'):
    """
    Generates a publication-quality plot of site populations over time.
    """
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 5))

    history = np.array(history)
    # Convert time to microseconds for better readability on the plot
    time_us = np.array(time_points) * 1e6

    for site_idx, label, style in sites_to_plot:
        ax.plot(time_us, history[:, site_idx], style, linewidth=2, label=label)

    ax.set_xlabel('Time (μs)', fontsize=14)
    ax.set_ylabel('Rydberg Population', fontsize=14)
    ax.set_title('Excitation Transport Dynamics', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, time_us[-1] if len(time_us) > 0 else 0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to '{filename}'")
    plt.close()

def animate_transport(Nx, coords, time_points, history, filename='transport_2d.mp4'):
    """
    Creates and saves a MP4 video of the transport with site labels.
    NOTE: This requires the 'ffmpeg' package to be installed on your system.
          You can install it via conda: `conda install -c conda-forge ffmpeg`
          or check the ffmpeg website for other installation options.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
                       xytext=(10, 10), 
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
    ani.save(filename, writer='ffmpeg', dpi=300, fps=20)
    print(f"Animation saved to '{filename}'")
    plt.close()


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. Simulation Setup ---
    # For larger systems (4x4 or bigger), use 'Schrodinger' mode to avoid memory issues
    # 'Lindblad' mode requires too much memory for systems with >12 atoms
    SIMULATION_MODE = 'Schrodinger'  # 'Schrodinger' or 'Lindblad'
    Nx, Ny = 7, 1
    
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
    r_v1, r_v2 = 11.4e-6, 12.8e-6
    r_h1, r_h2 = 10.0e-6, 13.8e-6
    
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
        'h1': Omega * 20, 'h2': Omega * 10,
        'v1': C6 / r_v1**6, 'v2': C6 / r_v2**6
    }
    delta_detuning_map = {
        'h1': -0.133 * Omega, 'h2': -0.033 * Omega,
        'v1': -0 * Omega, 'v2': -0.2 * Omega
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
    initial_site = 0
    target_site = 5  # 与第一个文件保持一致，目标到第6个原子
    
    # 模拟5个完整周期，每个周期包含2个脉冲
    total_periods = 5
    pulse_sequence = []
    for period in range(total_periods):
        pulse_sequence.extend([
            ('h1', pi_pulse_duration, 50),  # 第一个脉冲
            ('h2', pi_pulse_duration, 50),  # 第二个脉冲
        ])
    
    times, history = run_simulation(Nx, Ny, params, pulse_sequence, initial_site)

    # --- 5. Analyze and Visualize Results ---
    if history:
        final_populations = history[-1]
        fidelity = final_populations[target_site]
        print(f"\n--- Results ---")
        print(f"Transport Fidelity to site {target_site}: {fidelity:.4f}")

        sites_for_plot = [
            (initial_site, f'Initial Site ({initial_site})', 'r-'),
            (1, 'Intermediate Site (1)', 'g--'),
            (2, 'Intermediate Site (2)', 'c--'),
            (3, 'Intermediate Site (3)', 'm--'),
            (4, 'Intermediate Site (4)', 'y--'),
            (target_site, f'Target Site ({target_site})', 'b-')
        ]
        plot_transport_dynamics(times, history, sites_for_plot)
        animate_transport(Nx, atom_coords, times, history)
    else:
        print("\nSimulation did not produce data. Check parameters.")