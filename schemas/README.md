这四份 JSON Schema 描述 VeriCCL 的标准输入格式：topology 为拓扑，sketch 为集合通信与预算，atom 为合成策略和硬约束，runtime 为实际运行配置。字段定义来自 vericcl/input/schema.py。

文件结构检查后，load_request 继续检查算子语义、M/S 整除、rank 映射、预算和模块之间的约束。重复 JSON 键由读取器检查。JSON Schema 本身不执行这些跨字段检查，也不测量硬件能力。

字段省略或为 null 时，只有已定义默认值的字段才补入默认值。无默认值的必填字段仍需提供。S 可暂时未指定，由后续容量选择模块确定；N 始终由 M/S 推导。整数输入采用有符号 64 位表示范围，具体运行时字段还受各自更小的上限限制。

兼容写法先归一化，再检查标准结构。保留算子大小写和已列出的别名；支持的旧 solver 参数从 hyperparameters 移到 solver，旧策略开关从 atom 顶层移到 strategies。旧 total_solve_timeout_s 对应初始合成预算 max_initial_synthesis_s。同一设置同时出现在新旧位置时直接报错。input_chunkup、N、force_resolve、force_recalibrate 和三个旧合成布尔字段已移除。

value-sources.json 记录字段来自原文件、默认值、命令行还是推导结果。命令行可以补充未指定项；修改文件中的具体值必须显式使用 --override-input。输入文件本身保持不变。
