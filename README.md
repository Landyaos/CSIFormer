# CSIFormer: A Transformer-Based Channel Estimation and Equalization Framework

## 项目简介
本项目旨在构建一个基于 Transformer 的联合信道估计与均衡模型，主要应用于 5G 高速铁路通信和车联网 V2X 场景。项目核心功能包括：
- 数据仿真与处理
- 模型构建与训练
- 实验结果分析与可视化

---

## 项目目录结构
以下是本项目的主要目录结构：

```plaintext
.
├── data/                   # 数据相关文件
│   ├── raw/                # 原始仿真数据
│   ├── processed/          # 预处理后的数据
│   ├── datasets.py         # 数据加载和预处理脚本
│   └── simulation/         # 仿真生成数据模块
│       ├── matlab/         # MATLAB 仿真代码及数据
│       │   ├── models/     # 信道模型 MATLAB 脚本
│       │   └── results/    # MATLAB 仿真输出数据
│       ├── configs/        # 仿真参数配置文件
│       ├── convert.py      # MATLAB 数据格式转换脚本
│       └── generate_data.sh # 一键生成仿真数据的 Shell 脚本
├── models/                 # 模型相关代码
├── training/               # 训练与验证相关代码
├── experiments/            # 实验配置和结果
├── scripts/                # 实用脚本
├── tests/                  # 测试代码
├── notebooks/              # Jupyter/Colab notebooks
├── README.md               # 项目简介与目录结构说明
├── LICENSE                 # 开源协议
├── .gitignore              # Git忽略规则
├── requirements.txt        # Python依赖包列表
└── setup.py                # Python包配置
