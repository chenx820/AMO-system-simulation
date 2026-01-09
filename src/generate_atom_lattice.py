import numpy as np
import matplotlib.pyplot as plt
import os


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

    def plot_lattice(node_A_pos, node_B_pos, bond_list, node_A_id=None, node_B_id=None):
        """Plot the lattice using Matplotlib with atom labels."""
        
        # Extract all A and B node coordinates for plotting
        nodes_A = np.array(list(node_A_pos.values()))
        nodes_B = np.array(list(node_B_pos.values()))

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot edges
        def plot_edges(edges, color, label):
            # Add label only for the first edge to avoid legend duplication
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
        ax.axis('equal') # Ensure equal aspect ratio so angles are correct

        ax.legend(fontsize=14)

        lattice_path = f'outputs/plots/lattice/distorted_honeycomb_lattice_{N1}x{N2}.svg'
        os.makedirs(os.path.dirname(lattice_path), exist_ok=True)
        plt.savefig(lattice_path, bbox_inches='tight')
        plt.close()

    plot_lattice(node_A_pos, node_B_pos, bond_list, node_A_id, node_B_id)

    return node_A_pos, node_B_pos, bond_list, node_A_id, node_B_id