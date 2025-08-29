#!/usr/bin/env python3
"""
System Parameters Management Class
manage all physical parameters and system configurations of the AMO system

Author: Chen Huang
Date: 2025-08-22
"""

import numpy as np
import qutip as qt

class AMOSystemParams:
    """
    AMO system parameters management class
    manage all physical parameters of the AMO system
    """
    
    def __init__(self, Nx=4, Ny=4, Omega_MHz=3.0, simulation_mode='Schrodinger'):
        """
        initialize system parameters
        
        Args:
            Nx, Ny: atom array size
            Omega_MHz: Rabi frequency (MHz, linear frequency)
            simulation_mode: simulation mode ('Schrodinger' or 'Lindblad')
        """
        self.Nx = Nx
        self.Ny = Ny
        self.total_atoms = Nx * Ny
        self.hilbert_dim = 2**self.total_atoms
        
        # laser parameters (SI units)
        self.Omega = 2 * np.pi * Omega_MHz * 1e6  # angular frequency (rad/s)
        self.Omega_eff = self.Omega / np.sqrt(2)   # effective Rabi frequency
        self.pi_pulse_duration =0.5 * (2 * np.pi) / self.Omega_eff
        
        # decay and dephasing parameters
        self.Gamma_decay = 0.002 * self.Omega
        self.Gamma_dephasing = 0.004 * self.Omega
        
        # atom spacing (meters)
        self.r_h1, self.r_h2 = 11.4e-6, 12.8e-6
        self.r_v1, self.r_v2 = 10.8e-6, 14.2e-6
        
        # van der Waals coefficient
        self.C6 = 3e7 * self.Omega * (1e-6)**6  # (rad/s) * m^6
        
        # interaction strength map
        self.V_map = {
            'h1': self.C6 / self.r_h1**6,
            'h2': self.C6 / self.r_h2**6,
            'v1': self.C6 / self.r_v1**6,
            'v2': self.C6 / self.r_v2**6
        }
        
        # detuning map (initialized to 0, can be optimized by optimize_detuning.py)
        self.delta_detuning_map = {
            'h1': -0.133 * self.Omega,
            'h2': -0.033 * self.Omega,
            'v1': -0.166 * self.Omega,
            'v2': -0.016 * self.Omega
        }
        
        # simulation mode
        self.simulation_mode = simulation_mode
        
        # check memory requirements
        self._check_memory_requirements()
    
    def _check_memory_requirements(self):
        """check memory requirements and give warning"""
        if self.simulation_mode == 'Lindblad' and self.total_atoms > 12:
            print(f"WARNING: {self.total_atoms} atoms = {self.hilbert_dim} Hilbert dimension")
            print(f"Lindblad mode requires ~{self.hilbert_dim**2 * 16 / 1e9:.1f} GB memory")
            print("Switching to Schrodinger mode to avoid memory overflow...")
            self.simulation_mode = 'Schrodinger'
    
    def get_params_dict(self):
        """return parameter dictionary, used in run_simulation function"""
        return {
            'mode': self.simulation_mode,
            'Omega': self.Omega,
            'V_map': self.V_map,
            'Gamma_decay': self.Gamma_decay,
            'Gamma_dephasing': self.Gamma_dephasing,
            'delta_detuning_map': self.delta_detuning_map
        }
    
    def update_detuning(self, delta_h1=None, delta_h2=None, delta_v1=None, delta_v2=None):
        """
        update detuning parameters
        
        Args:
            delta_h1, delta_h2, delta_v1, delta_v2: new detuning values (relative to Omega)
        """
        if delta_h1 is not None:
            self.delta_detuning_map['h1'] = delta_h1 * self.Omega
        if delta_h2 is not None:
            self.delta_detuning_map['h2'] = delta_h2 * self.Omega
        if delta_v1 is not None:
            self.delta_detuning_map['v1'] = delta_v1 * self.Omega
        if delta_v2 is not None:
            self.delta_detuning_map['v2'] = delta_v2 * self.Omega
    
    def print_system_info(self):
        """print system information"""
        print("--- System Parameters ---")
        print(f"Grid: {self.Nx}x{self.Ny}, Mode: {self.simulation_mode}")
        print(f"Hilbert space dimension: 2^{self.total_atoms} = {self.hilbert_dim:,}")
        print(f"Omega: {self.Omega/(2*np.pi*1e6):.2f} MHz")
        print(f"Estimated pi-pulse duration: {self.pi_pulse_duration*1e6:.2f} us")
        
        # print interaction strength
        spacing_values = {'h1': self.r_h1, 'h2': self.r_h2, 'v1': self.r_v1, 'v2': self.r_v2}
        for key, val in self.V_map.items():
            r_val = spacing_values[key]
            print(f"V_{key} (r={r_val*1e6:.1f} um): {val/(2*np.pi*1e6):.2f} MHz")
        
        if self.simulation_mode == 'Lindblad':
            print(f"Gamma_decay: {self.Gamma_decay/(2*np.pi*1e3):.3f} kHz")
            print(f"Gamma_dephasing: {self.Gamma_dephasing/(2*np.pi*1e3):.3f} kHz")
    
    def get_optimized_solver_options(self, duration):
        """
        return optimized solver options based on system size
        
        Args:
            duration: pulse duration
            
        Returns:
            dict: solver options
        """
        if self.total_atoms <= 16:
            return {
                "store_final_state": True, 
                "nsteps": 10000, 
                "max_step": duration/1000,
                "rtol": 1e-8,
                "atol": 1e-8
            }
        else:
            return {
                "store_final_state": True, 
                "nsteps": 5000, 
                "max_step": duration/500,
                "rtol": 1e-8,
                "atol": 1e-8
            }
    
    def get_optimized_steps_per_pulse(self):
        """return optimized number of steps per pulse based on system size"""
        if self.total_atoms <= 16:
            return 25
        else:
            return 50
    
    def create_pulse_sequence(self, directions, steps_per_pulse=None):
        """
        create a pulse sequence
        
        Args:
            directions: pulse direction list, e.g. ['h1', 'v1', 'h2', 'v2']
            steps_per_pulse: number of steps per pulse, if None then use optimized value
            
        Returns:
            list: pulse sequence
        """
        if steps_per_pulse is None:
            steps_per_pulse = self.get_optimized_steps_per_pulse()
        
        return [(direction, self.pi_pulse_duration, steps_per_pulse) 
                for direction in directions]
    
    def get_system_size_info(self):
        """return system size information"""
        return {
            'Nx': self.Nx,
            'Ny': self.Ny,
            'total_atoms': self.total_atoms,
            'hilbert_dim': self.hilbert_dim,
            'memory_per_state_mb': self.hilbert_dim * 16 / 1e6
        }


# predefined system configurations
def create_3x3_system(Omega_MHz=3.0):
    """create a 3x3 system"""
    return AMOSystemParams(Nx=3, Ny=3, Omega_MHz=Omega_MHz)

def create_4x4_system(Omega_MHz=3.0):
    """create a 4x4 system"""
    return AMOSystemParams(Nx=4, Ny=4, Omega_MHz=Omega_MHz)

def create_custom_system(Nx, Ny, Omega_MHz=3.0, simulation_mode='Schrodinger'):
    """create a custom-sized system"""
    return AMOSystemParams(Nx=Nx, Ny=Ny, Omega_MHz=Omega_MHz, simulation_mode=simulation_mode)
