# VeriCCL

VeriCCL是一个面向GPU集群的集合通信调度生成与验证库。当前实现通过拓扑、sketch和atom三类输入生成MSCCL XML，并执行语义、时序、资源、BDD、模拟及可选在线验证。

## 项目特点

- 基于拓扑结构的通信优化
- 支持多种集体通信操作（如AllReduce, AllGather等）
- 与NCCL兼容的接口
- 灵活的通信调度器
- 高效的路由算法

## 安装

### 前提条件

- Python 3.9+
- 依赖包：
  - z3-solver
  - argcomplete
  - lxml
  - gurobipy
  - numpy
  - ply

### 安装步骤

```bash
# 安装依赖
pip install -e .
```

## 使用方法

VeriCCL提供以下公共命令：

```bash
vericcl solve --topology TOPOLOGY --sketch SKETCH --atoms ATOMS
vericcl verify --topology TOPOLOGY --sketch SKETCH --atoms ATOMS --xml XML
```

### 示例

规范输入位于`vericcl/examples/{topo,sketch,atom}`。迁移保留的旧输入位于`vericcl/examples/legacy`，历史模板位于`vericcl/examples/templates`。

详细的使用方法和参数说明请参考命令行帮助：

```bash
vericcl --help
vericcl solve --help
vericcl verify --help
```

## 性能测试

性能测试相关的代码和数据可以在`exp`目录下找到。

## 许可证

本项目使用MIT许可证。详情请参阅LICENSE文件。

```
Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
```
