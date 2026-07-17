# VeriCCL迁移说明

本仓库已将旧TACCL实现迁移为独立的`vericcl`包。公共命令统一为`vericcl solve`和`vericcl verify`，旧包名、旧命令及旧Python导入不再受支持。

## 路径变更

- 规范输入：`vericcl/examples/{topo,sketch,atom}`
- 旧拓扑和sketch：`vericcl/examples/legacy/{topo,sketch}`
- 旧参考XML：`vericcl/examples/legacy/Allgather.n16-1MB_i8_v1.xml`
- 历史模板：`vericcl/examples/templates`
- MSCCL跟踪补丁：`runtime/msccl-trace`

旧源码目录已在确认新包不依赖旧Python模块后删除。隔离测试会阻止任何旧包导入，并在同一进程中完成`solve`和`verify`。

## 兼容性与来源

旧拓扑转换器继续识别外部格式标识`LEGACY_TACCL_TOPOLOGY_FORMAT`。所有必须保留的旧标识由`vericcl.provenance.ALLOWED_TACCL_REFERENCES`集中说明，未列入允许清单的源代码引用会使测试失败。

旧示例、模板和参考XML仅用于格式转换、回归测试及来源追踪，不参与新工作流的运行时导入。完整迁移前历史可通过Git提交记录获取。
