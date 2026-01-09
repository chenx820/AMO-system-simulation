"""
AMO System Simulation Package

This package provides tools for simulating atomic manipulation and transport
in Rydberg atom arrays using quantum dynamics.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .system_params import AMOSystemParams, create_3x3_system, create_4x4_system, create_custom_system

__all__ = [
    "AMOSystemParams",
    "create_3x3_system", 
    "create_4x4_system",
    "create_custom_system"
]
