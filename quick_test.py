#!/usr/bin/env python3
"""
Quick performance test for 4x4 transport simulation
"""

import time
import numpy as np
import qutip as qt
from transport_2D import run_simulation, create_alternating_2d_array

def quick_4x4_test():
    """Quick test of 4x4 simulation performance"""
    
    print("=== Quick 4x4 Performance Test ===\n")
    
    # Setup parameters
    Nx, Ny = 4, 4
    total_atoms = Nx * Ny
    hilbert_dim = 2**total_atoms
    
    print(f"System: {Nx}x{Ny} = {total_atoms} atoms")
    print(f"Hilbert space dimension: 2^{total_atoms} = {hilbert_dim:,}")
    print(f"Estimated memory per state: {hilbert_dim * 16 / 1e6:.1f} MB")
    print()
    
    # Physical parameters
    Omega = 2 * np.pi * 3.0e6  # 3.0 MHz
    Omega_eff = Omega / np.sqrt(2)
    pi_pulse_duration = np.pi / Omega_eff
    
    # Spacings
    r_h1, r_h2 = 11.4e-6, 12.8e-6
    r_v1, r_v2 = 10.0e-6, 13.8e-6
    C6 = 3e7 * Omega * (1e-6)**6
    
    # Interaction map
    V_map = {
        'h1': C6 / r_h1**6, 'h2': C6 / r_h2**6,
        'v1': C6 / r_v1**6, 'v2': C6 / r_v2**6
    }
    delta_detuning_map = {
        'h1': -0 * Omega, 'h2': -0 * Omega,
        'v1': -0 * Omega, 'v2': -0 * Omega
    }
    
    params = {
        'mode': 'Schrodinger',
        'Omega': Omega,
        'V_map': V_map,
        'Gamma_decay': 0.002 * Omega,
        'Gamma_dephasing': 0.004 * Omega,
        'delta_detuning_map': delta_detuning_map
    }
    
    # Optimized pulse sequence
    steps_per_pulse = 25
    pulse_sequence = [
        ('h1', pi_pulse_duration, steps_per_pulse),
        ('h2', pi_pulse_duration, steps_per_pulse),
    ]
    
    print("Starting optimized simulation...")
    start_time = time.time()
    
    try:
        times, history = run_simulation(Nx, Ny, params, pulse_sequence, initial_sites=[0])
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        print(f"\n=== Results ===")
        print(f"Simulation completed in: {simulation_time:.2f} seconds")
        print(f"Total time points: {len(times)}")
        print(f"Time per step: {simulation_time/len(times)*1000:.2f} ms")
        
        if history:
            final_populations = history[-1]
            print(f"Final population at target site 2: {final_populations[2]:.4f}")
            
        return simulation_time, len(times)
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None

if __name__ == "__main__":
    quick_4x4_test()
