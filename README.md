# AMO System Simulation

A comprehensive Python package for simulating atomic manipulation and transport in Rydberg atom arrays using quantum dynamics. **Now with customizable pulse sequence management!**

## 📁 Project Structure

```
AMO-system-simulation/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── system_params.py         # Parameter management
│   ├── transport_2d.py          # 2D transport simulation
│   ├── optimize_detuning.py     # Detuning optimization
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── visualization.py    # Visualization tools (future)
├── notebooks/                   # Jupyter notebooks
│   └── transport_1d.ipynb      # 1D transport examples
├── outputs/                     # Output files
│   ├── plots/                   # Static images (PNG, PDF)
│   ├── animations/              # Video files (MP4, GIF)
│   └── data/                    # Data files (CSV, HDF5)
├── configs/                     # Configuration files
│   ├── default_params.json     # Default parameters
│   └── custom_pulse_sequences.json  # Custom pulse sequences
├── tests/                       # Test files
│   └── __init__.py
├── run_transport_2d.py          # Main simulation script
├── run_optimization.py          # Optimization script
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🚀 Quick Start

程序还没完善，模块还没集成完毕，可以先直接运行

```bash
python run_transport_2d.py
```

参数在`src/transport_2d.py`和`configs/default_params.json`手动修改。
