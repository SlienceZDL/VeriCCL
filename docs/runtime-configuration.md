# VeriCCL运行时配置

本文说明如何使一个XML step对应一个完整软件slice。硬件分包仍由网络和GPU运行时处理，不属于VeriCCL的软件粒度约束。

## 固定MSCCL编译参数

在MSCCL源码的`src/include/msccl.h`中确认以下常量：

```c
#define MSCCL_CHUNKSTEPS 4
#define MSCCL_SLICESTEPS 4
```

因此`SlicePerChunk = MSCCL_CHUNKSTEPS / MSCCL_SLICESTEPS = 1`。修改后在MSCCL源码根目录重新构建：

```bash
make -j src.build
```

VeriCCL trace补丁已固定并在communicator初始化时核对`MSCCL_CHUNKSTEPS 4`与`MSCCL_SLICESTEPS 4`。补丁应用、检查和时钟同步步骤见`runtime/msccl-trace/README.md`。

## 每次执行的环境变量

设任务的`slice_size_bytes=S`，必须使用：

```text
NCCL_BUFFSIZE=2*slice_size_bytes
```

例如`S=1048576`时，`NCCL_BUFFSIZE=2097152`。执行环境的核心约束为：

```bash
export NCCL_ALGO=MSCCL
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
```

一次执行只设置一个XML。VeriCCL不依赖MSCCL在同一communicator中注册或选择多个算法。

XML中的`cnt`固定为1，`slice_size_bytes`保持求解时的真实软件传输粒度。`NCCL_BUFFSIZE=2*S`配合`SlicePerChunk=1`，使MSCCL不再把一个step拆成多个软件slice；这不限制PCIe、NVLink、IB或以太网的硬件分包。

## 精确消息范围

XML使用Simple协议，并设置精确的`[minBytes,maxBytes)`范围。除AllGather外，`minBytes=total_size_bytes`且`maxBytes=minBytes+1`；AllGather按MSCCL接口使用`minBytes=rank_count*total_size_bytes`。`nccl-tests`必须传入与XML一致的唯一消息大小、数据类型、归约操作、root和原地模式。

若XML的TB step数、TB数量、channel数、依赖TB编号或偏移超过MSCCL限制，离线验证仍可生成`.candidate.xml`。报告中的`runtime_recommendations`会给出最小channel数和可整除的更大slice建议；该结果不可在线执行，建议只用于新一轮求解。

## release与trace运行

正式性能统计必须关闭trace：

```bash
export VERICCL_TRACE_ENABLE=0
```

逐step诊断使用单独的一次运行：

```bash
export VERICCL_TRACE_ENABLE=1
export VERICCL_TRACE_RECORDS=1048576
export VERICCL_TRACE_FILE_PREFIX=/absolute/path/to/vericcl-step
```

补丁将原始`iteration`字段写为MSCCL `workIndex`，用于区分重复NCCL集合调用。诊断命令固定使用`-w 0 -n 20 -c 0`；收集器丢弃每个计时块的setup调用，只分析请求模式对应的20次正式调用。`VERICCL_TRACE_RECORDS`是用户配置下限，预检会自动提升到至少`42 * max_steps_per_rank`。trace缓冲区溢出、缺少`s`或`r/rrc`端点、调用数异常、时钟同步失败或误差过大都会使在线算子验证失败，但不会否定离线语义与XML格式。trace运行开销不得计入release性能统计。

## `--online`前置条件

CLI从环境读取：

- `VERICCL_MSCCL_BUILD_DIR`：包含可加载MSCCL库的目录。
- `VERICCL_NCCL_TESTS_BUILD_DIR`：包含对应六类`nccl-tests`二进制的目录。
- `VERICCL_CLOCK_SYNC_BINARY`：已构建的GPU/MPI时钟同步工具。
- `VERICCL_ONLINE_INTER_NODE=0|1`：机内或机间执行。
- `VERICCL_MPI_LAUNCHER`和`VERICCL_MPI_HOSTFILE`：机间执行时必需。
- `VERICCL_MAX_CLOCK_UNCERTAINTY_US`：允许的时钟误差上限，默认10微秒。
- `VERICCL_CALIBRATION_LINK_CLASS=intra_node|inter_node`：本次校准的代表链路类别。
- `VERICCL_CALIBRATION_CACHE_PATH`：持久化校准点JSON的绝对路径。
- `VERICCL_GPU_MODEL`和`VERICCL_NIC_MODEL`：环境签名中的硬件型号。
- `VERICCL_CUDA_VERSION`、`VERICCL_NCCL_VERSION`和`VERICCL_MSCCL_VERSION`：环境签名中的软件版本。
- `VERICCL_FORCE_RECALIBRATE=0|1`：忽略匹配缓存并重新测试，默认0；输入中的`force_recalibrate=true`具有相同作用。
- `VERICCL_TRACE_RECORDS`：每Rank的trace记录容量下限，默认1048576；实际容量会按XML自动增大。

机内运行由一个`nccl-tests`进程使用`-g rank_count`启动全部本地GPU；机间运行由MPI为每个Rank启动一个`-g 1`进程。机间校准固定使用2个MPI进程，机内校准固定使用一个`-g 2`进程。算子与校准的launcher配置相互独立，选择机间校准不会使机内算子额外经过MPI启动。

正式统计每轮启动20个独立`nccl-tests`进程；每个进程内部使用5次预热和20次迭代。CV超过5%时最多执行3轮。报告保留每轮中位数、P95、均值、标准差、CV和稳定性，不选择最佳单次结果。

链路校准固定使用128 MiB和当前slice大小，机内仅测试1机2卡，机间仅测试2机1卡，并覆盖`k=1..min(max_calibration_channels,32,128MiB/S,link_max_channels)`。每个`k`生成独立Broadcast XML；完整wave的耗时由逐step trace计算，尾部不完整wave仍执行但不进入`D_safe(k)`。若`S`不能整除128 MiB，则校准状态为`not_run`且不改变slice大小。

环境签名精确覆盖链路类别、拓扑、GPU/NIC、CUDA/NCCL/MSCCL、Simple协议、slice大小、128 MiB、并发度、`NCCL_BUFFSIZE`、chunk/slice steps、GPU可见性、库路径及输入的`NCCL_*`和`UCX_*`变量；hostfile与NCCL拓扑文件还包含内容SHA-256。签名完全匹配时复用`VERICCL_CALIBRATION_CACHE_PATH`中的测量点；任一字段变化都会重新测试。缓存写入使用进程间文件锁、原子替换和`fsync`。

`solve --online`获得稳定校准后，保留每条链路或共享资源原有的`alpha`，只更新精确同构链路类别的`invbw`和`B_link(k)`，并将可用channel上限限制到最大实测并发度；随后重新构建Plan、二次求解并对新XML执行正式统计与trace验证。首轮和二次求解候选均保留在报告中，并通过`parent_candidate_id`关联。未选择的链路类别继续使用拓扑输入中的保守参数。`verify --online`不会静默改写现有XML；它在校准后继续验证当前XML，并在报告中设置`requires_resolve=true`，提示后续`solve`使用新参数重新求解。
