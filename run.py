#!/usr/bin/env python3
"""
Main script for running 2D transport simulation

Author: Chen Huang
Date: 2025-09-10
"""

import argparse

from src.transport_2d import run_tranport_2d


def main():
    parser = argparse.ArgumentParser(description='Run 2D transport simulation')
    parser.add_argument('--example', type=str, default='./examples/default.json', help='Path to example JSON to load')
        
    args = parser.parse_args()
    
    print(f"Running transport simulation... \n")
    
    run_tranport_2d(args.example)

if __name__ == '__main__':
    main()
