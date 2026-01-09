"""
Test lattice generation module for pulse optimization.

This module contains functions to generate and visualize a test lattice
used for optimizing quantum transport pulses in honeycomb structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import json
import os

os.makedirs('configs', exist_ok=True)   
os.makedirs('outputs/plots/test', exist_ok=True)

# define the single-atom operators
sm = qt.destroy(2) # lowering operator |0><1|
sp = qt.create(2)  # raising operator |1><0|
sz = qt.sigmaz()   # Pauli Z operator |0><0| - |1><1|
n_op = sp * sm  # number operator |1><1|

def generate_test_lattice(l1, l2, l3, theta1_deg, theta2_deg, theta3_deg, 
                         output_filename="outputs/plots/test/test_lattice.svg"):
    """
    Generate a test lattice for optimizing pulse.
    
    This function creates a minimal test lattice with one A-type node at the origin
    and three B-type nodes connected via r1, r2, and r3 bonds. This is useful for
    testing and optimizing pulse sequences before running full lattice simulations.
    
    Args:
        l1, l2, l3: Bond lengths (in micrometers)
        theta1_deg, theta2_deg, theta3_deg: Bond angles (in degrees)
        output_filename: Output filename for the saved figure (default: "test_lattice.svg")
    
    Returns:
        bond_list: List of bonds in format [(p1, p2, bond_type), ...]
    """
    # Convert angles from degrees to radians
    theta1 = np.deg2rad(theta1_deg)
    theta2 = np.deg2rad(theta2_deg)
    theta3 = np.deg2rad(theta3_deg)

    # Define the three basic bond vectors r1, r2, r3
    r1 = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)]) # red
    r2 = np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)]) # blue
    r3 = np.array([l3 * np.cos(theta3), l3 * np.sin(theta3)]) # brown

    # Define lattice vectors a1, a2 for A-type nodes
    a1 = r3 - r2
    a2 = r1 - r2
    p_B_offset = r3

    # Plot the lattice
    fig, ax = plt.subplots(figsize=(10, 8))

    pos_A = np.array([0, 0])
    ax.scatter(pos_A[0], pos_A[1], color='black', s=200, label='A node')

    n1n2_list = [(0, 0, 'r3'), (-1, 0, 'r1'), (-1, 1, 'r2')]
    bond_list = []
    for n1, n2, bond_type in n1n2_list:
        pos_B = n1 * a1 + n2 * a2 + p_B_offset
        bond_list.append((pos_A, pos_B, bond_type))

        ax.scatter(pos_B[0], pos_B[1], color='gray', s=200, label='B node' if n1n2_list.index((n1, n2, bond_type)) == 0 else '')

    # Plot edges
    color_map = {'r1': 'tab:red', 'r2': 'tab:blue', 'r3': 'tab:orange'}
    for (p1, p2, bond_type) in bond_list:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                color=color_map[bond_type], lw=4, label=bond_type, zorder=0)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    ax.set_xlabel('X coordinate (um)', fontsize=18)
    ax.set_ylabel('Y coordinate (um)', fontsize=18)
    ax.set_title('Test Lattice', fontsize=18)
    ax.legend(fontsize=14)

    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Test lattice saved to '{output_filename}'")
    
    return bond_list


def get_operator_for_site(op, i, N):
        """Create operator acting on site i in N-site system."""
        op_list = [qt.qeye(2)] * N
        op_list[i] = op
        return qt.tensor(op_list)


def build_hamiltonian(N, bond_list, Omega, current_delta, V_map): 
    """
    Build the system Hamiltonian.
    H = sum_j (Omega/2 * sigma_x_j - current_delta * n_j) + sum_{(i,j) in bonds} V_ij * n_i * n_j
    """
    H = 0
    
    # To improve efficiency, precompute all single-site operators
    sx_ops = [get_operator_for_site(qt.sigmax(), i, N) for i in range(N)]
    n_ops = [get_operator_for_site(n_op, i, N) for i in range(N)]

    # Driving term
    for i in range(N):
        H += 0.5 * Omega * sx_ops[i]
        H -= current_delta * n_ops[i]

    # Interaction terms (using precomputed bond list)
    # bond_list format: (i, j, _, _, bond_type) or (i, j, bond_type) or (p1, p2, bond_type)
    for bond in bond_list:
        if len(bond) == 5:
            # Format: (i, j, P_A, P_B, 'bond_type')
            i, j, _, _, bond_type = bond
        elif len(bond) == 3:
            # Format: (i, j, 'bond_type') or (p1, p2, 'bond_type')
            # Check if first two elements are integers (atom indices) or arrays (positions)
            if isinstance(bond[0], (int, np.integer)) and isinstance(bond[1], (int, np.integer)):
                i, j, bond_type = bond
            else:
                # Position format - skip, as we need atom indices
                # For test lattice, we'll create proper bond_list separately
                continue
        else:
            raise ValueError(f"Invalid bond format: {bond}. Expected (i, j, bond_type) or (i, j, P_A, P_B, bond_type)")
        
        # Get V_ij from V_map, default to 0 if bond type is not defined
        V_ij = V_map.get(bond_type, 0)
        
        if V_ij > 0:
            H += V_ij * (n_ops[i] * n_ops[j])
                
    return H



def simulate_test_lattice_with_laser(N, bond_list, params, pulse_sequence, 
                                     initial_site=0):
    """
    Simulate the test lattice with laser applied to the central atom (A node).
    Calculate population (probability in |1>) for each atom.
    
    The test lattice has 4 atoms:
    - Atom 0: A node at origin (central atom, where laser is applied)
    - Atom 1: B node connected via r3
    - Atom 2: B node connected via r1  
    - Atom 3: B node connected via r2
    
    Args:
        l1, l2, l3: Bond lengths (in micrometers)
        theta1_deg, theta2_deg, theta3_deg: Bond angles (in degrees)
        Omega: Rabi frequency (MHz)
        V_r1, V_r2, V_r3: Interaction strengths for each bond (MHz)
        pulse_sequence: Pulse sequence [(bond_type, duration, amplitude), ...]
        steps: Number of time steps (default: 100)
        initial_site: Site index to apply laser (default: 0, the central A node)
        pulse_shape: Custom pulse shape function f(t) -> amplitude, or string:
                    - None or 'constant': constant pulse (default)
                    - 'gaussian': Gaussian pulse
                    - 'square': Square pulse
                    - Custom function: f(t, args) -> amplitude
                    Function signature: f(t, args) where t is time, args is dict with keys:
                    {'t_max', 'Omega', 'center', 'width', etc.}
    
    Returns:
        times: Time points array
        populations: Dictionary with population (probability in |1>) for each atom {atom_id: population_array}
        final_state: Final quantum state
    """

    # Initial state: all atoms in |0> except initial_site in |1>
    initial_state_list = [qt.basis(2, 0)] * N
    initial_state_list[initial_site] = qt.basis(2, 1)
    psi0 = qt.tensor(initial_state_list)

    # Expectation operators (tracking Rydberg population)
    e_ops = [get_operator_for_site(n_op, i, N) for i in range(N)]

    options = {
            'store_final_state': True, 
            'nsteps': 5000, 
            'atol': 1e-8, 
            'rtol': 1e-6
        }

    # Time evolution - accumulate results over pulse sequence
    all_times = []
    all_populations = {i: [] for i in range(N)}
    
    for bond_type, duration, nsteps in pulse_sequence:
        t_max = params['T_pulse_unit'] * params['T_pulse_duration'][bond_type] * duration
        times = np.linspace(0, t_max, nsteps + 1)
        current_delta = params['V_map'][bond_type] + params['delta_detuning_map'][bond_type]
        H = build_hamiltonian(N, bond_list, params['Omega'], current_delta, params['V_map'])

        result = qt.sesolve(H, psi0, times, e_ops=e_ops, options=options)

        # Accumulate time points and populations
        if len(all_times) == 0:
            all_times.extend(times)
            for i in range(N):
                all_populations[i].extend(result.expect[i])
        else:
            # Skip first time point to avoid duplication
            all_times.extend(times[1:])
            for i in range(N):
                all_populations[i].extend(result.expect[i][1:])

        psi0 = result.final_state
    
    # Convert to numpy arrays
    populations = {i: np.array(all_populations[i]) for i in range(N)}
    times = np.array(all_times)

    return times, populations, result.final_state

def plot_population_evolution(times, populations, Omega=None, output_filename="test_lattice_population.svg"):
    """
    Plot population evolution (probability in |1>) for each atom.
    
    Args:
        times: Time points array
        populations: Dictionary with population for each atom {atom_id: population_array}
        Omega: Rabi frequency (for normalization, optional)
        output_filename: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize time by Omega/(2*pi) if Omega is provided
    if Omega is not None:
        times_plot = times / params['T_pulse_unit']
        xlabel = r'$\Omega t / (2\pi)$'
    else:
        times_plot = times
        xlabel = 'Time'
    
    # Plot population for each atom
    colors = ['black', 'tab:orange', 'tab:red', 'tab:blue']
    labels = ['Atom 0 (A, central)', 'Atom 1 (B, r3)', 'Atom 2 (B, r1)', 'Atom 3 (B, r2)']
    
    for i in range(len(populations)):
        if i not in populations or len(populations[i]) == 0:
            print(f"Warning: Skipping plot for {labels[i]} (no data)")
            continue
        
        # Ensure lengths match
        min_len = min(len(times_plot), len(populations[i]))
        ax.plot(times_plot[:min_len], populations[i][:min_len], 
               color=colors[i], lw=2, label=labels[i], 
               linestyle='-' if i == 0 else '--')
    
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(r'Population $\langle n_i \rangle$', fontsize=18)
    ax.set_title('Population Evolution: Laser on Central Atom', fontsize=18)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(times_plot[0], times_plot[-1])
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Population plot saved to '{output_filename}'")


def plot_population_heatmap(times, detunings, population_matrix, Omega=None,
                            atom_label='Atom 2 (B, r1)',
                            output_filename="test_lattice_population_heatmap.svg"):
    """
    Plot a 2D heatmap showing population of a specific atom versus detuning and time.
    """
    times_plot = times / params['T_pulse_unit']
    xlabel = r'$\tilde{\Omega} t / (2\pi)$'
    
    detunings = np.array(detunings)
    population_matrix = np.array(population_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(times_plot, detunings, population_matrix,
                         cmap='Blues', shading='auto')
    cbar = plt.colorbar(mesh, ax=ax, label=r'Population $\langle n_i \rangle$')

    # Find maximum population point
    max_idx = np.unravel_index(np.argmax(population_matrix), population_matrix.shape)
    max_detuning = detunings[max_idx[0]]
    max_time = times_plot[max_idx[1]]
    max_value = population_matrix[max_idx]
    print(f"{atom_label} max population {max_value:.4f} at detuning {max_detuning:.4f} MHz, normalized time {max_time:.4f}")
    
    ax.scatter(max_time, max_detuning, color='gold', edgecolor='black', s=80, zorder=5)
    ax.annotate(f"{max_value:.2f}",
                (max_time, max_detuning),
                textcoords="offset points", xytext=(6, 6),
                fontsize=10, color='black', weight='bold',
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="black", lw=0.5))
    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Detuning Δ (MHz)', fontsize=16)
    ax.set_title(f'{atom_label} Population vs Detuning and Time', fontsize=16)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Population heatmap saved to '{output_filename}'")


    return max_value, max_detuning, max_time

if __name__ == "__main__":

    # dimension of the lattice
    L1, THETA1_DEG = 11.5, 180.0  # um, deg
    L2, THETA2_DEG = 13.4, -70.0  # um, deg
    L3, THETA3_DEG = 10.5, 60.0   # um, deg

    Omega = 2 * np.pi * 0.3 # Rabi frequency, MHz
    Omega_eff = Omega / np.sqrt(2) # effective Rabi frequency
    T_pulse_unit = (2 * np.pi) / Omega_eff

    N = 4

    # van der Waals interaction strength
    C6 = 3e7 * Omega # MHz

    # van der Waals interaction strength
    V_r1 = C6 / L1**6  # r1, MHz
    V_r2 = C6 / L2**6  # r2, MHz
    V_r3 = C6 / L3**6  # r3, MHz

    

    params = {
    'Omega': Omega, # Rabi frequency, MHz
    'Omega_eff': Omega_eff, # effective Rabi frequency
    'T_pulse_unit': T_pulse_unit, 
    'mode': 'Schrodinger',    # 'Schrodinger' or 'Lindblad'
    'Gamma_decay': 0.002 * Omega,
    'Gamma_dephasing': 0.004 * Omega,
    'distance_map': {
        'r1': L1,
        'r2': L2,
        'r3': L3,
    },
    'V_map': {
        'r1': V_r1,
        'r2': V_r2,
        'r3': V_r3,
    },    
    'delta_detuning_map': {
        'r1': 0.0,
        'r2': 0.0,
        'r3': 0.0,
        },
    'T_pulse_duration': {
        'r1': 1.0,
        'r2': 1.0,
        'r3': 1.0
        }
    }
    
    # Generate test lattice visualization
    bond_list_vis = generate_test_lattice(
        L1, L2, L3, 
        THETA1_DEG, THETA2_DEG, THETA3_DEG,
        output_filename="outputs/plots/test/test_lattice.svg"
    )

    print(f"Generated {len(bond_list_vis)} bonds in test lattice")
    
    # Create bond_list with atom indices for simulation
    # Test lattice: Atom 0 (A) connects to Atom 1 (B, r3), Atom 2 (B, r1), Atom 3 (B, r2)
    bond_list = [
        (0, 1, None, None, 'r3'),  # Atom 0 -> Atom 1 via r3
        (0, 2, None, None, 'r1'),  # Atom 0 -> Atom 2 via r1
        (0, 3, None, None, 'r2'),  # Atom 0 -> Atom 3 via r2
    ]
    
    print(f"\n--- Quantum Simulation Parameters ---")
    print(f"Omega = {Omega/(2*np.pi):.3f} MHz")
    print(f"V_r1 = {V_r1:.3f} MHz")
    print(f"V_r2 = {V_r2:.3f} MHz")
    print(f"V_r3 = {V_r3:.3f} MHz")

    # Testing V_r1
    print(f"\n--- Testing V_r1 ---")

    base_pulse_sequence = [
        ('r1', 0.5, 200),
    ]
        
    delta_values = np.arange(-0.02, 0, 0.001)
    atom_to_track = 2  # Atom 2 (B, r1)
    population_matrix = []
        
    for delta_detuning in delta_values:
        params['delta_detuning_map']['r1'] = delta_detuning
        pulse_sequence = list(base_pulse_sequence)
            
        times, populations, final_state = simulate_test_lattice_with_laser(
                N, bond_list, params, pulse_sequence, initial_site=0
            )
            
        population_matrix.append(populations[atom_to_track])
        
    population_matrix = np.array(population_matrix)
        
    max_value, max_detuning, max_time = plot_population_heatmap(times, delta_values, population_matrix, Omega,
                                atom_label='Atom 2 (B, r1)',
                                output_filename="outputs/plots/test/atom2_population_heatmap.png")

    params['delta_detuning_map']['r1'] = max_detuning
    params['T_pulse_duration']['r1'] = max_time


    # Testing V_r2
    print(f"\n--- Testing V_r2 ---")

    base_pulse_sequence = [
        ('r2', 0.5, 200),
    ]

    delta_values = np.arange(-0.2, 0, 0.001)
    atom_to_track = 3  # Atom 3 (B, r2)
    population_matrix = []
    
    for delta_detuning in delta_values:
        params['delta_detuning_map']['r2'] = delta_detuning
        pulse_sequence = list(base_pulse_sequence)
        
        times, populations, final_state = simulate_test_lattice_with_laser(
            N, bond_list, params, pulse_sequence, initial_site=0
        )
        
        population_matrix.append(populations[atom_to_track])
    
    population_matrix = np.array(population_matrix)
    
    max_value, max_detuning, max_time = plot_population_heatmap(times, delta_values, population_matrix, Omega,
                            atom_label='Atom 3 (B, r2)',
                            output_filename="outputs/plots/test/atom3_population_heatmap.png")

    params['delta_detuning_map']['r2'] = max_detuning
    params['T_pulse_duration']['r2'] = max_time

    # Testing V_r3
    print(f"\n--- Testing V_r3 ---")

    base_pulse_sequence = [
        ('r3', 0.5, 200),
    ]

    delta_values = np.arange(-0.2, 0, 0.001)
    atom_to_track = 1  # Atom 1 (B, r3)
    population_matrix = []
    
    for delta_detuning in delta_values:
        params['delta_detuning_map']['r3'] = delta_detuning
        pulse_sequence = list(base_pulse_sequence)
        
        times, populations, final_state = simulate_test_lattice_with_laser(
            N, bond_list, params, pulse_sequence, initial_site=0
        )
        
        population_matrix.append(populations[atom_to_track])
    
    population_matrix = np.array(population_matrix)
    
    max_value, max_detuning, max_time = plot_population_heatmap(times, delta_values, population_matrix, Omega,
                            atom_label='Atom 1 (B, r3)',
                            output_filename="outputs/plots/test/atom1_population_heatmap.png")

    params['delta_detuning_map']['r3'] = max_detuning
    params['T_pulse_duration']['r3'] = max_time
    
    # Write the parameters to a JSON file
    with open('configs/optimized_params.json', 'w') as f:
        json.dump(params, f)


