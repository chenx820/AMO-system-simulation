import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import matplotlib.animation as animation
import os
import json

# from generate_atom_lattice import generate_distorted_honeycomb

# define the single-atom operators
sm = qt.destroy(2) # lowering operator |0><1|
sp = qt.create(2)  # raising operator |1><0|
sz = qt.sigmaz()   # Pauli Z operator |0><0| - |1><1|
n_op = sp * sm  # number operator |1><1|

# lattice size
N1, N2 = 5, 5
    
# bond lengths and angles
THETA1_DEG = 180.0 # deg
THETA2_DEG = -70.0 # deg
THETA3_DEG = 60.0  # deg

OUTPUT_PLOTS_DIR = os.path.join('outputs', 'plots')
OUTPUT_ANIM_DIR = os.path.join('outputs', 'animations')
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANIM_DIR, exist_ok=True)

CONFIG_PATH = os.path.join('configs','optimized_params.json')


def load_params(config_path=CONFIG_PATH):
    """Load parameters from config if available, otherwise use defaults."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            params = json.load(f)
    return params

def get_operator_for_site(op, i, N):
    """
    Creates a full Hilbert space operator for a single-qubit operator 'op' acting on site 'i',
    """
    op_list = [qt.qeye(2)] * N
    op_list[i] = op
    return qt.tensor(op_list)
    

def build_hexagon_hamiltonian(local_n, Omega, current_delta, V_map, bond_types_in_hex): 
    """
    Build Hamiltonian for a single hexagon (6 atoms max).
    
    Parameters:
    -----------
    local_n : int
        Number of atoms in this hexagon (5 or 6)
    bond_types_in_hex : list of (local_i, local_j, bond_type)
        Bonds within the hexagon using local indices
    """
    H = 0
    
    # Precompute operators for local indices
    sx_ops = [get_operator_for_site(qt.sigmax(), i, local_n) for i in range(local_n)]
    n_ops = [get_operator_for_site(n_op, i, local_n) for i in range(local_n)]
    
    # Driving term for all atoms in hexagon
    for i in range(local_n):
        H += 0.5 * Omega * sx_ops[i]
        H -= current_delta * n_ops[i]

    # Interaction terms within hexagon
    for (local_i, local_j, bond_type) in bond_types_in_hex:
        V_ij = V_map.get(bond_type, 0)
        if V_ij > 0:
            H += V_ij * (n_ops[local_i] * n_ops[local_j])
                
    return H


def get_collapse_operators_local(local_n, Gamma_decay_val, Gamma_dephasing_val):
    """
    Build collapse operators for local hexagon
    """
    c_ops = []
    for k in range(local_n):
        c_ops.append(np.sqrt(Gamma_decay_val) * get_operator_for_site(sm, k, local_n))
    for k in range(local_n):
        c_ops.append(np.sqrt(Gamma_dephasing_val) * get_operator_for_site(sz, k, local_n))
    return c_ops


def get_hexagon_bonds(laser_atoms, bond_list):
    """
    Get bonds within a hexagon, converted to local indices.
    
    Returns:
    --------
    list of (local_i, local_j, bond_type)
    """
    # 创建全局索引到局部索引的映射
    valid_atoms = [a for a in laser_atoms if a != -1]
    global_to_local = {g: l for l, g in enumerate(valid_atoms)}
    
    bonds_in_hex = []
    for (i, j, _, _, bond_type) in bond_list:
        if i in global_to_local and j in global_to_local:
            bonds_in_hex.append((global_to_local[i], global_to_local[j], bond_type))
    
    return bonds_in_hex


def animate_transport(N_sites, coords_map, bond_list, time_points, history, filename=None):
    """
    Create an MP4 animation of the honeycomb lattice population evolution.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    history_norm = np.array(history)
    vmax = np.max(history_norm) if np.max(history_norm) > 0 else 1.0

    coords_display = np.array([coords_map[i] for i in range(N_sites)])
    color_map = {'r1': 'tab:red', 'r2': 'tab:blue', 'r3': 'tab:orange'}
    
    # coordinate range
    xlim = (coords_display[:, 0].min() - 5, coords_display[:, 0].max() + 5)
    ylim = (coords_display[:, 1].min() - 5, coords_display[:, 1].max() + 5)

    def update(frame):
        ax.clear()
        
        # Plot static lattice bonds (zorder=1, background)
        for (i, j, p1, p2, bond_type) in bond_list:
            color = color_map.get(bond_type, 'gray')
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                     color=color, lw=1.0, zorder=1, alpha=0.5)

        # Plot dynamic populations (zorder=3, foreground)
        populations = history_norm[frame]
        colors = populations
        # (keep the original scaling logic)
        sizes = 2000 * populations + 50 
        
        ax.scatter(coords_display[:, 0], coords_display[:, 1], 
                   c=colors, s=sizes, cmap='viridis', 
                   vmin=0, vmax=vmax, zorder=3,
                   edgecolors='black', linewidth=0.5)
        
        # Plot atom index labels (zorder=4)
        for i in range(N_sites):
            ax.annotate(f"{i}", 
                       (coords_display[i, 0], coords_display[i, 1]),
                       ha='center', va='center',
                       fontsize=8,
                       color='black',
                       zorder=4)
        
        # Set axes and title
        ax.set_title(f'Honeycomb Excitation Transport\nTime: {time_points[frame]:.2f} μs') 
        ax.set_xlabel('X position (μm)')
        ax.set_ylabel('Y position (μm)')
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Create and save animation
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50)
    if filename is None:
        filename = os.path.join(OUTPUT_ANIM_DIR, 'honeycomb_transport_animation.mp4')
    ani.save(filename, writer='ffmpeg', dpi=300, fps=20) 
    print(f"Animation saved to '{filename}'")
    plt.close()

def run_simulation(N_total, bond_list, params, pulse_sequence, hexagon_dict, initial_site): 
    """
    Run time evolution in local hexagon subspace (6 atoms each pulse).
    
    Parameters:
    -----------
    N_total : int
        Total number of atoms in the full lattice (for output)
    pulse_sequence : list of tuples
        Each tuple: (bond_type, laser_hexagon_id, (src, tgt), steps)
    hexagon_dict : dict
        {hexagon_id: [atom indices]}
    initial_site : int
        Initial excited site (global index)
    """
    
    # key: global_atom_index, value: current population (0 or 1 initially)
    populations = {i: 0.0 for i in range(N_total)}
    populations[initial_site] = 1.0
    
    history = []
    time_points = []
    t_total = 0.0
    
    print(f"Start simulation: {params['mode']} mode, using local hexagon subspace")
    
    for pulse_idx, pulse in enumerate(pulse_sequence):
        direction, laser_hex_id, (src, tgt), steps = pulse
        
        # Get atoms in this hexagon
        laser_atoms = hexagon_dict[laser_hex_id]
        valid_atoms = [a for a in laser_atoms if a != -1]
        local_n = len(valid_atoms)
        
        # global to local index mapping
        global_to_local = {g: l for l, g in enumerate(valid_atoms)}
        local_to_global = {l: g for l, g in enumerate(valid_atoms)}
        
        # find excited state in local space
        if src not in global_to_local:
            raise ValueError(f"Source atom {src} not in hexagon {laser_hex_id}: {valid_atoms}")
        local_excited = global_to_local[src]
        
        # build local initial state
        initial_state_list = [qt.basis(2, 0)] * local_n
        initial_state_list[local_excited] = qt.basis(2, 1)
        psi0 = qt.tensor(initial_state_list)
        
        # local expectation operators
        e_ops = [get_operator_for_site(n_op, i, local_n) for i in range(local_n)]
        
        # get bonds in hexagon
        bonds_in_hex = get_hexagon_bonds(laser_atoms, bond_list)
        
        duration = params['T_pulse_unit'] * params['T_pulse_duration'][direction]
        current_delta = params['V_map'][direction] + params['delta_detuning_map'][direction]
        
        H = build_hexagon_hamiltonian(local_n, params['Omega'], current_delta, params['V_map'], bonds_in_hex)
        H_func = lambda t, args: H
        
        t_pulse = np.linspace(t_total, t_total + duration, steps + 1)
        
        if params['mode'] == 'Schrodinger':
            result = qt.sesolve(H_func, psi0, t_pulse, e_ops=e_ops, options={'progress_bar': 'tqdm', 'store_final_state': True})
        elif params['mode'] == 'Lindblad':
            rho0 = psi0 * psi0.dag() if psi0.isket else psi0
            c_ops = get_collapse_operators_local(local_n, params['Gamma_decay'], params['Gamma_dephasing'])
            result = qt.mesolve(H_func, rho0, t_pulse, c_ops=c_ops, e_ops=e_ops, options={'progress_bar': 'tqdm', 'store_final_state': True})
        else:
            raise ValueError(f"Invalid simulation mode: {params['mode']}")
        
        # map local result back to global
        local_history = np.array(result.expect).T  # shape: (steps+1, local_n)
        
        for t_idx in range(len(result.times)):
            if pulse_idx == 0 or t_idx > 0:  # avoid duplicate first time point
                time_points.append(result.times[t_idx])
                
                # build global population snapshot
                global_pop = np.zeros(N_total)
                for local_i in range(local_n):
                    global_i = local_to_global[local_i]
                    global_pop[global_i] = local_history[t_idx, local_i]
                history.append(global_pop)
        
        # Print target population at end of this pulse
        if tgt in global_to_local:
            local_tgt = global_to_local[tgt]
            final_pop = local_history[-1, local_tgt]
            print(f"Pulse {pulse_idx+1}: L{laser_hex_id} ({direction}) | {src} -> {tgt} | Target pop = {final_pop:.4f}")
        else:
            print(f"Pulse {pulse_idx+1}: L{laser_hex_id} ({direction}) | {src} -> {tgt} | Target {tgt} not in hexagon!")
        
        t_total += duration

    time_period_points = np.array(time_points) / params['T_pulse_unit']
    return time_points, time_period_points, np.array(history)


def get_hexagon_atoms(N1, N2):
    """
    Get the atom indices for each hexagon ring in the honeycomb lattice.
    
    A hexagon consists of 6 positions (in ring order):
    A(n1, n2), B(n1, n2), A(n1+1, n2-1), B(n1, n2-1), A(n1, n2-1), B(n1-1, n2)
    
    If a node is out of bounds, its index is set to -1.
    
    Returns:
        dict: {hexagon_id: [6 atom indices, -1 for missing nodes]}
    """
    hexagon_dict = {}
    hex_id = 0
    A_base = N1 * N2  # offset for B nodes
    
    for n1 in range(N1):
        for n2 in range(N2):
            nodes = [
                ('A', n1, n2),
                ('B', n1, n2),
                ('A', n1+1, n2-1),
                ('B', n1, n2-1),
                ('A', n1, n2-1),
                ('B', n1-1, n2),
            ]
            
            # calculate atom indices, out-of-bounds set to -1
            atom_indices = []
            for node_type, i, j in nodes:
                if i < 0 or i >= N1 or j < 0 or j >= N2:
                    atom_indices.append(-1)
                elif node_type == 'A':
                    atom_indices.append(i * N2 + j)
                else:  # B
                    atom_indices.append(A_base + i * N2 + j)
            
            # only keep hexagons with at least 5 valid atoms
            valid_count = sum(1 for idx in atom_indices if idx != -1)
            if valid_count >= 5:
                hexagon_dict[hex_id] = atom_indices
                hex_id += 1
    
    return hexagon_dict


def generate_distorted_honeycomb(N1, N2, l1, l2, l3, theta1_deg, theta2_deg, theta3_deg):
    """
    Generate the coordinates and edges of a distorted honeycomb lattice.

    Args:
    N1, N2: Lattice sizes along the two lattice vector directions
    l1, l2, l3: Lengths of the three bonds
    theta1_deg, theta2_deg, theta3_deg: Angles of the three bonds (in degrees)
    """

    # Convert angles from degrees to radians
    theta1 = np.deg2rad(theta1_deg)
    theta2 = np.deg2rad(theta2_deg)
    theta3 = np.deg2rad(theta3_deg)

    # Define the three basic bond vectors r1, r2, r3
    r1 = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)]) # red
    r2 = np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)]) # blue
    r3 = np.array([l3 * np.cos(theta3), l3 * np.sin(theta3)]) # brown

    print("--- Vector definitions ---")
    print(f"r1: {r1} (length: {l1}, angle: {theta1_deg} deg)")
    print(f"r2: {r2} (length: {l2}, angle: {theta2_deg} deg)")
    print(f"r3: {r3} (length: {l3}, angle: {theta3_deg} deg)")

    # Define lattice vectors a1, a2 for A-type nodes
    a1 = r3 - r2
    a2 = r1 - r2

    # Define the offset of B-type nodes relative to A-type nodes
    p_B_offset = r3

    node_A_pos = {} # A node coordinates dictionary { (n1,n2): [x,y] }
    node_B_pos = {} # B node coordinates dictionary { (n1,n2): [x,y] }
    node_A_id = {} # A node id dictionary { (n1,n2): i }
    node_B_id = {} # B node id dictionary { (n1,n2): i }

    coords = {}     # coordinate mapping {linear index i: [x, y]}

    # Generate all node coordinates
    current_i = 0

    # A node
    for n1 in range(N1):
        for n2 in range(N2):
            pos_A = n1 * a1 + n2 * a2
            node_A_pos[(n1, n2)] = pos_A
            node_A_id[(n1, n2)] = current_i
            coords[current_i] = pos_A
            current_i += 1
            
    # B node
    for n1 in range(N1):
        for n2 in range(N2):
            pos_B = n1 * a1 + n2 * a2 + p_B_offset
            node_B_pos[(n1, n2)] = pos_B
            node_B_id[(n1, n2)] = current_i
            coords[current_i] = pos_B
            current_i += 1

    N = current_i # total number of sites

    # Generate all edges (connections)
    # Each A node A(n1, n2) connects to three B nodes
    bond_list = []

    for (n1, n2), i in node_A_id.items():
        P_A = node_A_pos[(n1, n2)]
        
        # r3: A(n1, n2) --r3--> B(n1, n2)
        idx_B = (n1, n2)
        if idx_B in node_B_pos:
            j = node_B_id[idx_B]
            P_B = node_B_pos[idx_B]
            bond_list.append((i, j, P_A, P_B, 'r3'))

        # r1: A(n1, n2) --r1--> B(n1-1, n2+1)
        idx_B = (n1 - 1, n2 + 1)
        if idx_B in node_B_pos:
            j = node_B_id[idx_B]
            P_B = node_B_pos[idx_B]
            bond_list.append((i, j, P_A, P_B, 'r1'))

        # r2: A(n1, n2) --r2--> B(n1-1, n2)
        idx_B = (n1 - 1, n2)
        if idx_B in node_B_pos:
            j = node_B_id[idx_B]
            P_B = node_B_pos[idx_B]
            bond_list.append((i, j, P_A, P_B, 'r2'))

    def plot_lattice(node_A_pos, node_B_pos, bond_list, node_A_id=None, node_B_id=None, hexagon_dict=None, coords=None):
        """Plot the lattice using Matplotlib with atom labels and laser circles."""
        
        # Extract all A and B node coordinates for plotting
        nodes_A = np.array(list(node_A_pos.values()))
        nodes_B = np.array(list(node_B_pos.values()))

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot laser circles for hexagons (background)
        if hexagon_dict is not None and coords is not None:
            import matplotlib.patches as patches
            colors_hex = plt.cm.tab20(np.linspace(0, 1, len(hexagon_dict)))
            for hex_id, atom_indices in hexagon_dict.items():
                # get valid atom coordinates
                valid_coords = [coords[idx] for idx in atom_indices if idx != -1]
                if len(valid_coords) >= 5:
                    # calculate hexagon center
                    center = np.mean(valid_coords, axis=0)
                    # calculate radius (distance to farthest atom + some margin)
                    distances = [np.linalg.norm(c - center) for c in valid_coords]
                    radius = max(distances) + 1.5
                    # draw circle
                    circle = patches.Circle(center, radius, 
                                          fill=False, 
                                          edgecolor=colors_hex[hex_id % len(colors_hex)],
                                          linestyle='--', 
                                          linewidth=2, 
                                          alpha=0.7,
                                          zorder=1)
                    ax.add_patch(circle)
                    # label laser number
                    ax.text(center[0], center[1] + radius + 1, f'L{hex_id}',
                           ha='center', va='bottom', fontsize=9, 
                           color=colors_hex[hex_id % len(colors_hex)], weight='bold')
        
        # Plot edges
        def plot_edges(edges, color, label):
            plotted_label = False
            for (p1, p2) in edges:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=color, lw=4, 
                        label=label if not plotted_label else "")
                plotted_label = True

        edges_r1 = [bond_list[i][2:4] for i in range(len(bond_list)) if bond_list[i][4] == 'r1']
        edges_r2 = [bond_list[i][2:4] for i in range(len(bond_list)) if bond_list[i][4] == 'r2']
        edges_r3 = [bond_list[i][2:4] for i in range(len(bond_list)) if bond_list[i][4] == 'r3']

        plot_edges(edges_r1, 'tab:red', 'r1')
        plot_edges(edges_r2, 'tab:blue', 'r2')
        plot_edges(edges_r3, 'tab:orange', 'r3') 

        # Plot nodes (after edges so nodes appear on top)
        ax.scatter(nodes_A[:, 0], nodes_A[:, 1], c='black', s=200, zorder=3, label='A nodes')
        ax.scatter(nodes_B[:, 0], nodes_B[:, 1], c='gray', s=200, zorder=3, label='B nodes')

        # Add atom labels
        if node_A_id is not None:
            for (n1, n2), pos in node_A_pos.items():
                atom_id = node_A_id[(n1, n2)]
                ax.text(pos[0], pos[1], f'{atom_id}', 
                    fontsize=8, ha='center', va='center', 
                    color='white', weight='semibold', zorder=4)
        
        if node_B_id is not None:
            for (n1, n2), pos in node_B_pos.items():
                atom_id = node_B_id[(n1, n2)]
                ax.text(pos[0], pos[1], f'{atom_id}', 
                    fontsize=8, ha='center', va='center', 
                    color='white', weight='semibold', zorder=4)

        ax.set_xlabel(r'X coordinate [$\mu$m]', fontsize=18)
        ax.set_ylabel(r'Y coordinate [$\mu$m]', fontsize=18)
        ax.set_title(f'Distorted Honeycomb Lattice ({N1}x{N2})', fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axis('equal')

        ax.legend(fontsize=14)

        lattice_path = f'outputs/plots/lattice/distorted_honeycomb_lattice_{N1}x{N2}.svg'
        os.makedirs(os.path.dirname(lattice_path), exist_ok=True)
        plt.savefig(lattice_path, bbox_inches='tight')
        plt.close()

    # get hexagon dictionary
    hexagon_dict = get_hexagon_atoms(N1, N2)
    plot_lattice(node_A_pos, node_B_pos, bond_list, node_A_id, node_B_id, hexagon_dict, coords)

    return node_A_pos, node_B_pos, bond_list, node_A_id, node_B_id


if __name__ == "__main__":

    # Load parameters
    params = load_params()

    L1 = params['distance_map']['r1']
    L2 = params['distance_map']['r2']
    L3 = params['distance_map']['r3']

    hexagon_dict = get_hexagon_atoms(N1, N2)

    # Generate coordinates and edges
    node_A_pos, node_B_pos, bond_list, node_A_id, node_B_id = generate_distorted_honeycomb(
        N1, N2, L1, L2, L3, THETA1_DEG, THETA2_DEG, THETA3_DEG
        )

    print(f"\n--- Coordinate generation completed ---")
    print(f"Generated {len(node_A_pos)} A nodes in total.")
    print(f"Generated {len(node_B_pos)} B nodes in total.")


    coords_map = {}
    for (n1, n2), i in node_A_id.items():
        coords_map[i] = node_A_pos[(n1, n2)]
    for (n1, n2), i in node_B_id.items():
        coords_map[i] = node_B_pos[(n1, n2)]

    # Pulse sequence: (bond_type, laser_hexagon_id, (src, tgt), steps)
    pulse_sequence = [
        ('r3', 1, (1, 26), 25),
        ('r2', 1, (26, 6), 25),
        ('r3', 5, (6, 31), 25),
        ('r2', 5, (31, 11), 25) 
    ]

    # Extract initial site and traffic sequence from pulse_sequence
    initial_site = pulse_sequence[0][2][0]  # first pulse's src
    traffic_sequence = [initial_site]
    for pulse in pulse_sequence:
        traffic_sequence.append(pulse[2][1])  # each pulse's tgt

    times, times_period, history = run_simulation(
        N_total = N1 * N2 * 2,
        bond_list = bond_list,
        params = params,
        pulse_sequence = pulse_sequence,
        hexagon_dict = hexagon_dict,
        initial_site = initial_site
    )

    # ---- Plot population and pulse sequence ----
    fig, (ax_pop, ax_pulse) = plt.subplots(
        2, 1, figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Population evolution
    for site in traffic_sequence:
        ax_pop.plot(
            times_period,
            history[:, site],
            label=f'Site {site}',
            lw=4
        )

    ax_pop.set_ylabel(r'Population $\langle n_i\rangle$', fontsize=18)
    ax_pop.legend(loc='best', fontsize=14)
    ax_pop.grid(True, linestyle='--', alpha=0.6)

    # Pulse sequence (colored time blocks)
    color_map = {'r1': 'tab:red', 'r2': 'tab:blue', 'r3': 'tab:orange'}
    t_start = 0.0
    for direction, laser_hex, (src, tgt), steps in pulse_sequence:
        duration_norm = params['T_pulse_duration'][direction]
        t_end = t_start + duration_norm
        ax_pulse.axvspan(
            t_start, t_end,
            color=color_map.get(direction, 'gray'),
            alpha=0.3,
            edgecolor='none',
        )
        ax_pulse.text(
            0.5 * (t_start + t_end),
            0.5,
            f'{direction}\nL{laser_hex}',
            ha='center',
            va='center',
            fontsize=10
        )
        t_start = t_end

    ax_pulse.set_yticks([])
    ax_pulse.set_xlabel(r'$\tilde{\Omega}t/(2\pi)$', fontsize=18)
    ax_pulse.set_ylabel('Pulses', fontsize=14)
    ax_pulse.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    population_path = os.path.join(OUTPUT_PLOTS_DIR, 'population_evolution.svg')
    plt.savefig(population_path, bbox_inches='tight')
    plt.close()


    print("\n--- Generating animation ---")
    animate_transport(
        N_sites = N1 * N2 * 2,
        coords_map = coords_map,
        bond_list = bond_list,
        time_points = times,
        history = history,
        filename = os.path.join(OUTPUT_ANIM_DIR, 'honeycomb_transport_animation.mp4')
    )



    """
    TO DO:
    - 4 atoms (Benz)
    - 12 atoms (Benzene ring)

    - QEC
    - scale up
    """