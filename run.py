#!/usr/bin/env python3
"""
Simple runner script for AMO simulations

Usage:
    python run.py transport    # Run 2D transport simulation
    python run.py optimize     # Run detuning optimization
"""

import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [transport|optimize]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "transport":
        print("Running 2D transport simulation...")
        subprocess.run([sys.executable, "transport_2d.py"], cwd="src")
    elif command == "optimize":
        print("Running detuning optimization...")
        # Get command line arguments for optimization
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        cmd = [sys.executable, "optimize_detuning.py"] + args
        subprocess.run(cmd, cwd="src")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: transport, optimize")
        sys.exit(1)

if __name__ == '__main__':
    main()
