# Canvas: Scalable and Optimal Collective Communication Scheduling for Large-Scale GPU Clusters

Canvas是一个拓扑感知的集体通信库，专为高性能计算和分布式机器学习环境设计。它能够基于特定的网络拓扑结构优化通信模式，从而提高分布式系统的通信效率。

## 项目特点

- 基于拓扑结构的通信优化
- 支持多种集体通信操作（如AllReduce, AllGather等）
- 与NCCL兼容的接口
- 灵活的通信调度器
- 高效的路由算法

## 安装

### 前提条件

- Python 3.6+
- 依赖包：
  - z3-solver
  - argcomplete
  - lxml
  - gurobipy
  - numpy
  - ply

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/Sibuge/Canvas.git
cd Canvas

# 安装依赖
pip install -e .
```

## 使用方法

TACCL提供了命令行工具，可以通过以下命令使用：

```bash
# 生成通信方案
taccl solve [options]

# 合并通信方案
taccl combine [options]

# 转换为NCCL兼容格式
taccl ncclize [options]

# 搜索通信方案
taccl search [options]

# NCCL管道优化
taccl ncclize-pipeline [options]
```

### 示例

TACCL包含了一些示例，可以在`taccl/examples`目录下找到：
- 拓扑示例（`taccl/examples/topo/`）
- 通信方案示例（`taccl/examples/sketch/`）

详细的使用方法和参数说明请参考命令行帮助：

```bash
taccl --help
taccl <command> --help
```

## 性能测试

性能测试相关的代码和数据可以在`exp`目录下找到。

## 许可证

本项目使用MIT许可证。详情请参阅LICENSE文件。

```
Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
