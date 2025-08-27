#!/usr/bin/env python3
"""
Main script for running 2D transport simulation

Usage:
    python run_transport_2d.py --Nx 4 --Ny 4 --Omega 3.0
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run 2D transport simulation')
    parser.add_argument('--Nx', type=int, default=4, help='Number of atoms in x direction')
    parser.add_argument('--Ny', type=int, default=4, help='Number of atoms in y direction')
    parser.add_argument('--Omega', type=float, default=3.0, help='Rabi frequency in MHz')
    parser.add_argument('--mode', choices=['Schrodinger', 'Lindblad'], default='Schrodinger',
                       help='Simulation mode (Schrodinger or Lindblad)')
    
    args = parser.parse_args()
    
    print(f"Running {args.Nx}x{args.Ny} transport simulation...")
    print(f"Omega: {args.Omega} MHz, Mode: {args.mode}")
    
    # Import and run the simulation
    import subprocess
    import sys
    
    # Run the transport_2d.py script with proper path
    cmd = [sys.executable, 'src/transport_2d.py']
    subprocess.run(cmd, cwd=os.path.dirname(__file__))

if __name__ == '__main__':
    main()
