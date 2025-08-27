# 自定义脉冲序列使用指南

## 概述

本系统提供了一个灵活的脉冲序列管理系统，允许您通过JSON配置文件定义和运行自定义的原子传输脉冲序列。每个序列可以包含：

- **系统配置**：网格大小、激光参数等
- **传输路径**：起始点、目标点、中间点
- **脉冲序列**：方向、持续时间、失谐参数
- **优化设置**：是否启用优化、目标保真度等

## 文件结构

```
configs/
├── default_params.json          # 默认系统参数
└── custom_pulse_sequences.json  # 自定义脉冲序列配置

src/
├── pulse_manager.py             # 脉冲序列管理器
├── system_params.py             # 系统参数管理
├── transport_2d.py              # 2D传输模拟
└── optimize_detuning.py         # 失谐优化

run_custom_pulse.py              # 运行自定义序列的主脚本
examples/
└── add_custom_sequence.py       # 添加自定义序列的示例
```

## 快速开始

### 1. 查看可用序列

```bash
python run_custom_pulse.py --list
```

### 2. 查看序列详细信息

```bash
python run_custom_pulse.py --info basic_transport
```

### 3. 运行基本传输序列

```bash
python run_custom_pulse.py --sequence basic_transport
```

### 4. 运行优化的序列

```bash
python run_custom_pulse.py --sequence stepwise_optimized
```

## 配置文件格式

### 基本结构

```json
{
  "pulse_sequences": {
    "sequence_name": {
      "description": "序列描述",
      "system_config": {
        "Nx": 4,
        "Ny": 4,
        "Omega_MHz": 3.0,
        "simulation_mode": "Schrodinger"
      },
      "transport_path": {
        "initial_point": 0,
        "target_point": 15,
        "intermediate_points": [5, 10],
        "path_description": "传输路径描述"
      },
      "pulse_sequence": {
        "directions": ["h1", "v1", "h2", "v2"],
        "steps_per_pulse": 25,
        "custom_durations": null,
        "custom_detunings": null
      },
      "optimization": {
        "enabled": false,
        "target_fidelity": 0.95
      }
    }
  }
}
```

### 配置字段说明

#### system_config
- `Nx, Ny`: 网格大小
- `Omega_MHz`: 激光拉比频率 (MHz)
- `simulation_mode`: 模拟模式 ("Schrodinger" 或 "Lindblad")

#### transport_path
- `initial_point`: 起始点索引 (0-15 for 4x4网格)
- `target_point`: 目标点索引
- `intermediate_points`: 中间点列表 (可选)
- `path_description`: 路径描述

#### pulse_sequence
- `directions`: 脉冲方向列表
  - `h1`: 水平方向1 (短间距)
  - `h2`: 水平方向2 (长间距)
  - `v1`: 垂直方向1 (短间距)
  - `v2`: 垂直方向2 (长间距)
- `steps_per_pulse`: 每个脉冲的时间步数
- `custom_durations`: 自定义脉冲持续时间 (可选)
- `custom_detunings`: 自定义失谐参数 (可选)

#### optimization
- `enabled`: 是否启用优化
- `target_fidelity`: 目标保真度
- `optimization_params`: 优化参数 (可选)

## 网格索引映射

### 4x4 网格
```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

### 3x3 网格
```
0 1 2
3 4 5
6 7 8
```

## 自定义脉冲序列示例

### 1. L形路径传输

```json
{
  "custom_path_1": {
    "description": "L形路径传输",
    "system_config": {
      "Nx": 4, "Ny": 4, "Omega_MHz": 3.0
    },
    "transport_path": {
      "initial_point": 0,
      "target_point": 12,
      "intermediate_points": [3, 7, 11],
      "path_description": "L形路径：先向右再向下"
    },
    "pulse_sequence": {
      "directions": ["h1", "h1", "v1", "v1"],
      "steps_per_pulse": 20,
      "custom_durations": [0.5, 0.5, 0.5, 0.5],
      "custom_detunings": {
        "h1": [-0.15, -0.12],
        "v1": [-0.18, -0.15]
      }
    },
    "optimization": {
      "enabled": false,
      "target_fidelity": 0.90
    }
  }
}
```

### 2. 高保真度传输

```json
{
  "high_fidelity_transport": {
    "description": "高保真度传输",
    "system_config": {
      "Nx": 4, "Ny": 4, "Omega_MHz": 3.0
    },
    "transport_path": {
      "initial_point": 0,
      "target_point": 15,
      "intermediate_points": [5, 10],
      "path_description": "高保真度对角线传输"
    },
    "pulse_sequence": {
      "directions": ["h1", "v1", "h2", "v2"],
      "steps_per_pulse": 40,
      "custom_durations": null,
      "custom_detunings": null
    },
    "optimization": {
      "enabled": true,
      "target_fidelity": 0.99,
      "optimization_params": {
        "delta_range": 0.2,
        "coarse_steps": 30,
        "fine_window": 0.03,
        "fine_steps": 25
      }
    }
  }
}
```

## 通过代码添加序列

您也可以通过Python代码动态添加新的脉冲序列：

```python
from src.pulse_manager import PulseSequenceManager

# 创建管理器
manager = PulseSequenceManager()

# 定义新序列
new_sequence = {
    "description": "我的自定义序列",
    "system_config": {
        "Nx": 4, "Ny": 4, "Omega_MHz": 3.0
    },
    "transport_path": {
        "initial_point": 0,
        "target_point": 15,
        "intermediate_points": [5, 10],
        "path_description": "对角线传输"
    },
    "pulse_sequence": {
        "directions": ["h1", "v1", "h2", "v2"],
        "steps_per_pulse": 25
    },
    "optimization": {
        "enabled": false,
        "target_fidelity": 0.95
    }
}

# 添加序列
manager.add_custom_sequence("my_sequence", new_sequence)

# 运行序列
results = manager.run_sequence("my_sequence")
print(f"保真度: {results['fidelity']:.4f}")
```

## 命令行选项

### run_custom_pulse.py

```bash
# 基本用法
python run_custom_pulse.py --sequence <序列名>

# 列出所有序列
python run_custom_pulse.py --list

# 查看序列信息
python run_custom_pulse.py --info <序列名>

# 指定配置文件
python run_custom_pulse.py --config <配置文件路径>

# 不保存结果
python run_custom_pulse.py --sequence <序列名> --no-save
```

## 输出结果

运行序列后，系统会：

1. **显示模拟进度**：实时显示脉冲执行进度
2. **输出结果摘要**：包括保真度、目标点布居等
3. **保存结果文件**：在 `outputs/custom_sequences/<序列名>/` 目录下

### 结果文件格式

```json
{
  "sequence_name": "basic_transport",
  "fidelity": 0.9234,
  "target_population": 0.9234,
  "transport_path": {
    "initial_point": 0,
    "target_point": 15,
    "path_description": "对角线传输"
  },
  "pulse_sequence": [
    ["h1", 0.5236, 25],
    ["v1", 0.5236, 25],
    ["h2", 0.5236, 25],
    ["v2", 0.5236, 25]
  ],
  "timestamp": "20250127_143022"
}
```

## 优化参数说明

当启用优化时，系统会使用以下参数：

- `delta_range`: 失谐搜索范围 (相对于Omega)
- `coarse_steps`: 粗搜索步数
- `fine_window`: 精细搜索窗口
- `fine_steps`: 精细搜索步数

## 故障排除

### 常见问题

1. **配置文件未找到**
   ```bash
   # 确保配置文件存在
   ls configs/custom_pulse_sequences.json
   ```

2. **序列不存在**
   ```bash
   # 查看可用序列
   python run_custom_pulse.py --list
   ```

3. **配置格式错误**
   ```bash
   # 验证JSON格式
   python -m json.tool configs/custom_pulse_sequences.json
   ```

4. **内存不足**
   - 减少网格大小 (Nx, Ny)
   - 使用Schrodinger模式而不是Lindblad模式
   - 减少steps_per_pulse

### 性能优化

- 对于大系统 (>16个原子)，建议使用Schrodinger模式
- 优化序列通常需要更多计算时间
- 可以通过调整steps_per_pulse来平衡精度和速度

## 扩展功能

### 添加新的脉冲方向

如果需要添加新的脉冲方向，需要修改 `system_params.py` 中的失谐映射：

```python
self.delta_detuning_map = {
    'h1': -0.133 * self.Omega,
    'h2': -0.033 * self.Omega,
    'v1': -0.166 * self.Omega,
    'v2': -0.016 * self.Omega,
    'new_direction': -0.1 * self.Omega  # 新方向
}
```

### 自定义优化算法

可以通过继承 `PulseSequenceManager` 类来实现自定义的优化算法。

## 联系和支持

如有问题或建议，请联系开发团队。
