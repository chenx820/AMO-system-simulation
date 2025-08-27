#!/usr/bin/env python3
"""
Main script for running detuning optimization

Usage:
    python run_optimization.py --Nx 3 --Ny 3 --initial_site 0 --target_site 2
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run detuning optimization')
    parser.add_argument('--Nx', type=int, default=3, help='Number of atoms in x direction')
    parser.add_argument('--Ny', type=int, default=3, help='Number of atoms in y direction')
    parser.add_argument('--initial_site', type=int, default=0, help='Initial site index')
    parser.add_argument('--target_site', type=int, default=2, help='Target site index')
    parser.add_argument('--periods', type=int, default=1, help='Number of pulse periods')
    parser.add_argument('--steps_per_pulse', type=int, default=20, help='Time steps per pulse')
    parser.add_argument('--Omega', type=float, default=3.0, help='Rabi frequency in MHz')
    
    args = parser.parse_args()
    
    print(f"Running optimization for {args.Nx}x{args.Ny} system...")
    print(f"Transport: site {args.initial_site} → site {args.target_site}")
    
    # Import and run the optimization
    from src.optimize_detuning import main as optimize_main
    optimize_main(vars(args))

if __name__ == '__main__':
    main()
