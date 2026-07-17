# 基于形式化验证的真实执行集合通信库VeriCCL

## 1.Introduction

为了同时优化集合通信延迟和吞吐，现有方法通常采用step调度、flow调度或在线诊断。step调度具有较细的时间粒度，但在显式表达完整路径依赖后求解复杂度较高；flow调度便于优化稳态吞吐，但较粗的chunk及阶段同步会限制流水重叠；在线诊断能够观测实际执行，但通常不直接生成语义正确的调度修复。理想拓扑参数与实际设备状态之间的偏差还会进一步降低离线调度的实际性能。VeriCCL使用atom级路径与时间建模、组合求解、形式化语义验证、BDD机会分析、动态事件模拟和在线step诊断生成并调优候选。系统明确区分当前验证候选中的`selected_best`与具有求解器证明的`proven_optimal`，不默认承诺全局最优。

## 2.模块设计

VeriCCL采用新的`vericcl` Python包和`vericcl`命令行入口。现有TACCL代码中的拓扑、路由、调度和XML生成逻辑作为迁移基础，但通过明确接口拆分为`input`、`semantics`、`topology`、`planner`、`solver`、`composer`、`xml`、`verification`、`tuning`和`artifacts`模块。

`input`将三个JSON输入规范化为不可变规范；`semantics`提供slice、AggregateState、算子和逻辑缓冲区的唯一语义；`topology`建模单向链路、channel、共享资源及校准参数；`planner`生成全局PlanDAG和局部求解任务；`solver`执行构造式及MILP求解；`composer`完成局部阶段的全局语义合成；`xml`负责EndpointAtom、物理缓冲区、TB及MSCCL XML降低；`verification`执行语义验证、BDD机会分析、事件模拟、XML和在线验证；`tuning`生成候选级TuningOverlay；`artifacts`管理输出、报告、trace和缓存。

统一数据流为“JSON输入→规范化规范→全局PlanDAG→局部求解任务→逻辑atom调度→全局合成→语义验证/BDD机会分析/事件模拟→EndpointAtom及缓冲区降低→TB调度及XML验证→XML和报告→可选在线校准→更新拓扑并重新求解”。任何候选只有通过当前阶段要求的验证后才能进入下一阶段；在线验证只更新性能参数，不改变集合通信语义、slice定义或用户约束。

代码迁移时，现有包名、模块名、类名、函数名、命令行入口、日志前缀、缓存版本、生成文件名、示例命令和文档中的内部`taccl`标识应尽可能统一修改为`vericcl`。`setup.py`或后续构建配置必须安装`vericcl`分发包并注册`vericcl`命令，所有内部绝对导入同步修改。旧构建目录、`taccl.egg-info`及其他生成产物不得作为源代码继续使用。若重命名影响脚本、模板、测试或执行命令，必须在同一模块修改中同步修复，并在本文档、README和迁移说明中记录。

MSCCL/NCCL协议字段、外部文件格式中的固定schema名称、第三方版权和明确的TACCL来源说明不属于内部标识，只有在确认外部解析器和调用方均不受影响时才允许修改。例如MSCCL兼容格式要求保留的`sccl_type`不得仅为统一命名而改变。确需保留的`taccl`字符串必须在代码审查中逐项说明保留原因，不得遗漏或静默保留。

### (1)Atom建模
atom = [s, pt, t] = [(slice_id, slice_size), (stage_num, [stage_id, operat_num, operator, operat_list], ..., [stage_id, send_num, send_list]), (st_time, ed_time)]

VeriCCL使用atom建模集合通信中最细粒度的传输事件：其中s、pt和t都是向量，s包含slice的ID和大小；pt包含stage_num和若干stage，stage_num表示该slice从源Rank传输到当前Rank经历的stage数量，每个stage包括stage_id、operat_num、operator和operat_list。operat_num表示该stage的操作数量，operator表示SEND或REDUCE，operat_list通过`symbol[src, dst, ready_time]`记录执行路径，其中ready_time表示slice已经完整到达src并可执行下一操作的时间。t包含st_time和ed_time，分别表示当前操作开始和完整结束时间，ed_time通常成为后继symbol的ready_time。

设全局Rank数为`P`，每个源Rank具有固定的`N`个slice，则`slice_id = source_rank * N + logical_slice_index`，`source_rank = floor(slice_id / N)`，`logical_slice_index = slice_id mod N`。每个slice具有唯一源Rank。pt只记录该slice从初始源Rank到当前Rank的路径前缀；求解完成后，源Rank到任一当前Rank只保留一条确定的单阶段或多阶段链路。

求解器和验证器内部使用`PayloadState(rank, logical_address, contributors, ready_time, active, member_paths)`。非归约数据仍由原始slice ID区分；AggregateState的logical_address固定为logical_slice_index，contributors记录已经归约的完整slice ID集合，不创建新的外部slice ID。状态是否完整由当前PlanDAG节点要求的贡献集合决定：Broadcast、AllGather和AllToAll要求单一源贡献，Reduce、AllReduce和ReduceScatter要求相同逻辑位置的全部Rank贡献。

REDUCE的两个输入必须具有相同逻辑位置且contributors不相交；输出contributors为集合并集，输入状态版本被消费，目标位置产生新版本。同一状态版本不能作为多次REDUCE的源，同一Rank和逻辑归约位置最多存在一个活动AggregateState。目标Rank已有的本地贡献直接参与状态合并，不生成自环传输。AggregateState物理大小保持为一个slice_size，链路容量只计算一次，ready_time取全部输入完成时间的最大值。

完整状态可以按算子语义分支发送。不完整归约状态最多执行一次出站操作，发送后原状态失活，但发送前仍可接收并继续归约。AggregateState继续传输时，每个成员slice保留自身路径，公共后缀在外部atom中按成员展开；求解器、容量模型、BDD和XML通过同一transfer_id将其视为一次物理操作。若任一成员命中用户禁用项`(slice_id, src, dst, stage_id)`，该公共物理传输不可用。

#### Atom与MSCCL真实传输粒度

VeriCCL使用等长slice，不支持末尾短slice、填充slice或将一个atom拆分为多个XML step。设每个Rank的输入数据量为`total_size_bytes = M`，真实传输粒度为`slice_size_bytes = S`，则必须满足：

\[
M \bmod S = 0, \qquad N = M/S
\]

其中`N`是每个源Rank的slice数量。求解器、XML生成器和MSCCL运行时必须共同保证：

\[
\text{one atom}=\text{one XML step}=\text{one complete MSCCL software transfer unit}
\]

本定义不考虑PCIe、NVLink、InfiniBand或其他物理链路的硬件分包。

##### 1. VeriCCL输入参数

用户在`sketch.json`的`hyperparameters`中设置`total_size_bytes`和`slice_size_bytes`。`input_chunkup`由VeriCCL根据`N = total_size_bytes / slice_size_bytes`自动推导；如果用户同时提供`input_chunkup`，其值必须与`N`一致。所有大小均使用字节为单位。

`sketch.json`中的`CollectiveSpec`是集合通信语义的唯一规范化来源，包含`operator`、`root`、`datatype`、`reduction_op`和`inplace`。全局Rank数从`topo.json`推导，不允许在多个输入中重复定义；`inplace`默认值为`false`。Broadcast和Reduce必须提供合法`root`，Reduce、AllReduce和ReduceScatter必须提供受支持的`reduction_op`。Scatter和Gather可以作为内部计划节点使用，但不接受为独立求解目标。

```json
{
  "collective": {
    "operator": "allreduce",
    "root": null,
    "datatype": "float32",
    "reduction_op": "sum",
    "inplace": false
  }
}
```

命令行中的集合通信语义参数默认只用于补充缺失值或检查一致性。若用户需要覆盖`CollectiveSpec`，必须显式启用语义覆盖；规范化后的完整CollectiveSpec必须写入求解结果、验证报告和缓存签名。未启用覆盖时，命令行值与文件值不一致属于输入错误。

```json
{
  "hyperparameters": {
    "total_size_bytes": 268435456,
    "slice_size_bytes": 1048576,
    "input_chunkup": 256
  }
}
```

上例表示每个Rank的输入数据量为256 MiB，每个slice为1 MiB，因此每个源Rank有256个slice。输入解析阶段必须验证`M > 0`、`S > 0`和`M % S == 0`，不满足时直接报错，不自动调整数据量或分片大小。ReduceScatter和AllToAll还必须满足`N % P == 0`，其中`P`为全局Rank数。

`slice_size_bytes`是VeriCCL支持的最细粒度完整软件传输单元，在一次求解、验证和迭代调优任务内为不可变参数。在线和离线调优只能调整路径、channel、并发度、阶段组合和调度时间，不得修改`slice_size_bytes`或重新编号slice。

##### 2. XML参数

XML生成器固定使用`proto="Simple"`，每个物理atom生成一个`cnt="1"`的step，禁止把连续地址合并为`cnt > 1`的step。`srcoff`、`dstoff`和scratch offset均以`S`为单位，对应的字节偏移为`offset * S`。

XML生成器禁止生成`rcs`、`rrs`和`rrcs`，也禁止MSCCL将多个本地`re`操作融合为一个指令。中继路径必须生成独立的接收step和后继发送step，并使用`depid/deps`表达完整slice到达后的依赖。生成器仅使用`s`、`r`、`rrc`、`cpy`和`nop`：`s/r`用于SEND，`s/rrc`用于REDUCE，`cpy`仅用于本地缓冲区转换，`nop`仅用于表达跨TB依赖。

求解器保持缓冲区无关，只处理slice、AggregateState、Rank、路径、channel和时间。全局调度合成完成后，XML模块根据CollectiveSpec和`inplace`模式生成确定性的BufferPlan：

```text
ValueKey:
  RawValue(slice_id)
  AggregateValue(logical_slice_index, contributors, state_version)

PhysicalRef:
  rank
  buffer
  offset
  valid_from
  valid_until

BufferPlan:
  value_locations
  aliases
  local_copies
  i_chunks
  o_chunks
  s_chunks
```

AggregateState的物理值不能仅以`logical_slice_index`标识，因为同一逻辑位置在归约过程中会依次产生不同contributors集合的状态。`state_version`只用于区分内部状态及其活跃期，不改变外部slice定义。scratch采用确定性的活跃区间分配，只有活跃区间不相交的ValueKey才能复用同一offset。

设源Rank为`r`、逻辑位置为`l`、`q=N/P`，六类算子的最终输出地址为：

| 算子 | 最终物理输出 |
| --- | --- |
| Broadcast | 每个Rank的`o[l]` |
| Reduce | root的`o[l]` |
| AllGather | 每个Rank的`o[rN+l]` |
| AllReduce | 每个Rank的`o[l]` |
| AllToAll | 目标Rank `floor(l/q)`的`o[rq+(l mod q)]` |
| ReduceScatter | owner `floor(l/q)`的`o[l mod q]` |

所有普通输入slice在其源Rank的逻辑输入位置均为`i[l]`。非原地模式必须保持输入数据不被归约覆盖：原始贡献从`i[l]`读取；若输入需要成为本地归约累加器，先用显式`cpy`复制到最终output位置或scratch；网络接收只在活跃期安全时直接写入最终output，否则写入scratch；最终值未通过网络操作落到output时，再生成显式`cpy`。顶层`<copy>`标签不得承担运行时数据搬运语义。

原地模式由CollectiveSpec和具体算子契约共同确定，不使用统一的“输入offset等于输出offset”规则：Broadcast和AllReduce的`i[l]`与`o[l]`别名；Reduce仅在root上将`i[l]`与`o[l]`别名；AllGather在Rank `r`上将`i[l]`与`o[rN+l]`别名；ReduceScatter在Rank `r`上将最终`o[j]`与输入区域`i[rq+j]`别名；AllToAll的输入和输出共享应用缓冲区，接收可能覆盖尚未发送的slice时，必须先将仍然活跃且存在覆盖风险的输入复制到scratch。这些别名只影响XML物理地址，不改变求解器的逻辑位置。

每个显式`cpy`作为LocalOpNode进入依赖图，并放入`send=-1, recv=-1`的本地TB。BufferPlan生成后必须执行BufferLivenessVerifier，检查所有读取均发生在初始化或写入之后、`rrc`执行前接收端累加器已经存在、活跃ValueKey没有错误复用同一物理位置、原地别名不会覆盖尚未发送的数据、非原地输入不会被修改、最终contributors和输出地址满足CollectiveSpec，以及`i_chunks`、`o_chunks`和`s_chunks`覆盖全部实际offset。完整BufferPlan、别名关系、scratch峰值和每个显式复制的原因必须写入逐XML验证报告。

公开atom定义保持不变。XML生成器在内部使用端点atom封装：

```text
EndpointAtom = {
  atom,
  transfer_id,
  xml_type,
  rank,
  peer,
  channel
}
```

每次物理传输使用唯一`transfer_id`，并生成一对端点atom：SEND生成源Rank的`s`端点和目标Rank的`r`端点，REDUCE生成源Rank的`s`端点和目标Rank的`rrc`端点。两个端点共享逻辑传输信息和传输时间区间，只有双方均满足依赖并到达各自TB队首时才允许开始传输。求解器、容量约束和BDD按`transfer_id`将端点对视为一次物理传输，避免重复计算链路容量和路径。

XML通信TB严格使用单向结构。通信TB按`(rank, direction, peer, channel)`划分，每个通信TB只能包含发送端点或接收端点，以及表达该TB消费依赖所需的`nop`；即使目标Rank和channel相同，`s`与`r/rrc`也必须位于不同通信TB。`cpy`使用独立本地TB。XML生成前必须联合安排端点对在两侧通信TB中的顺序，并执行死锁检测；在满足语义依赖和无死锁的前提下，最大限度保持求解器给出的时间顺序。

端点排序采用同步列表调度。XML生成器首先根据`transfer_id`将配对的`s`与`r/rrc`端点临时收缩为一个`TransferNode`，并根据路径依赖、AggregateState汇合依赖、缓冲区依赖和跨TB依赖建立有向无环图。只有语义前驱均已调度的`TransferNode`才能进入就绪集合。每次选中一个节点后，生成器必须同时将两个端点追加到对应的发送TB和接收TB，保证同一有向链路和channel上的发送顺序与接收顺序一致。

当一个后继操作依赖多个REDUCE贡献时，选择`ed_time`最大者作为直接`depid/deps`前驱；若并列，则依次使用较长剩余关键路径和稳定标识确定。其余前驱各生成一个`nop`并置于消费该结果的通信TB中，`nop`分别依赖对应的前驱，TB串行顺序保证后继操作等待全部贡献。若同一结果由多个TB消费，每个消费TB独立生成汇合`nop`，不得通过一个TB的局部顺序隐式约束另一个TB。

就绪节点依次按照求解器给出的较小`st_time`、较长剩余关键路径、较小`ed_time`以及稳定标识`(stage_id, logical_slice_index, src, dst, channel, transfer_id)`排序。同步调度完成后，将每个TB内相邻step的串行关系加入依赖图并再次检查无环性。若出现依赖环，只允许交换相互之间不存在语义依赖的节点，并以相对求解调度顺序的逆序数量最少为修复目标；无法修复时必须拒绝生成XML并重新求解。

XML生成后必须执行端点级事件模拟。每个TB只能暴露其队首step；一次传输只有在配对的`s`与`r/rrc`同时位于各自TB队首且两侧依赖均满足时才能执行。如果仍有未完成step但不存在可执行的端点对，则判定为死锁并拒绝该XML。事件模拟属于XML正确性的强制验证，不得由在线性能验证替代。

设`P`为全局Rank数，各算子的XML chunk参数为：

| 算子 | `nchunksperloop` | `i_chunks` | `o_chunks` |
| --- | ---: | ---: | ---: |
| Broadcast | `N` | `N` | `N` |
| Reduce | `N` | `N` | `N` |
| AllReduce | `N` | `N` | `N` |
| AllToAll | `N` | `N` | `N` |
| ReduceScatter | `N` | `N` | `N/P` |
| AllGather | `P*N` | `N` | `P*N` |

每个XML只适用于生成该调度时使用的固定`P`、`M`、`S`、算子和原地/非原地模式。为避免MSCCL将同一XML用于其他消息大小，生成器设置精确的左闭右开区间：

```text
minBytes = runtime_bytes
maxBytes = runtime_bytes + 1
```

其中AllGather的`runtime_bytes = P * M`，Broadcast、Reduce、AllReduce、AllToAll和ReduceScatter的`runtime_bytes = M`。例如，8个Rank、`M = 256 MiB`、`S = 1 MiB`的AllReduce关键XML字段如下：

```xml
<algo name="vericcl" proto="Simple" coll="allreduce" ngpus="8" nchannels="1" nchunksperloop="256" minBytes="268435456" maxBytes="268435457" inplace="1">
  <gpu id="0" i_chunks="256" o_chunks="256" s_chunks="0">
    <tb id="0" send="1" recv="-1" chan="0">
      <step s="0" type="s" srcbuf="o" srcoff="0" dstbuf="o" dstoff="0" cnt="1" depid="-1" deps="-1" hasdep="0"/>
    </tb>
  </gpu>
</algo>
```

##### 3. MSCCL编译期参数

`MSCCL_CHUNKSTEPS`和`MSCCL_SLICESTEPS`定义在MSCCL源码的`src/include/msccl.h`中。用户必须将两者都设为4，使`SlicePerChunk = 1`：

```c
#define MSCCL_CHUNKSTEPS 4
#define MSCCL_SLICESTEPS 4
```

修改后需要在MSCCL根目录重新编译：

```sh
cd /path/to/msccl
make -j src.build
```

默认构建产物位于`<msccl-root>/build`。运行VeriCCL XML时，应确保`LD_LIBRARY_PATH`指向该定制MSCCL构建，避免加载系统中的其他NCCL/MSCCL库。

##### 4. MSCCL运行时参数

`NCCL_BUFFSIZE`是运行时环境变量，必须在启动通信进程和执行`ncclCommInitRank`之前设置。固定使用：

\[
\text{NCCL\_BUFFSIZE}=2S
\]

例如，`S = 1 MiB`时：

```sh
export NCCL_BUFFSIZE=2097152
export LD_LIBRARY_PATH=/path/to/msccl/build/lib:${LD_LIBRARY_PATH}
export MSCCL_XML_FILES=/path/to/vericcl.xml
export NCCL_ALGO=MSCCL
```

使用MPI时，必须把变量传递到所有Rank：

```sh
mpirun -np 8 \
  -x NCCL_BUFFSIZE=2097152 \
  -x LD_LIBRARY_PATH=/path/to/msccl/build/lib:${LD_LIBRARY_PATH} \
  -x MSCCL_XML_FILES=/path/to/vericcl.xml \
  -x NCCL_ALGO=MSCCL \
  /path/to/application
```

`NCCL_BUFFSIZE`属于communicator级参数。同一communicator内使用的VeriCCL XML必须具有相同的`S`；需要改变`S`时，必须重新设置`NCCL_BUFFSIZE`并重建communicator。

##### 5. 粒度验证

生成XML前和在线试运行前必须验证：

1. 协议为Simple，且每个通信step的`cnt = 1`。
2. `MSCCL_CHUNKSTEPS = MSCCL_SLICESTEPS = 4`。
3. `NCCL_BUFFSIZE = 2 * slice_size_bytes`。
4. `(args.count * sizeMultiplier * datatype_size_bytes) / nchunksperloop = slice_size_bytes`，且不存在余数。
5. `chunkSize = NCCL_BUFFSIZE / 2 = slice_size_bytes`。
6. XML的`minBytes/maxBytes`与当前运行数据量完全匹配。
7. MSCCL线程数、数据类型和`chunkSize`满足运行时对齐约束。

调度或XML自身违反atom粒度语义时，VeriCCL必须拒绝生成该XML，不得自动拆分、合并或填充atom。仅涉及本机MSCCL编译参数、运行时环境或执行容量限制时，不影响逻辑调度生成、BDD机会分析和离线调优，但在实际执行前必须通过运行时预检查。

##### 6. MSCCL执行兼容性检查

当前MSCCL限制每个TB最多包含256个step，每个channel最多包含32个发送TB和32个接收TB，每个Rank最多包含216个TB，并且最多使用32个channel。通信step、`cpy`和`nop`均计入TB的step数量。此外，`srcoffset`、`dstoff`和归约源offset使用16位有符号整数，因此所有非负XML缓冲区offset均不得超过32767；`dependentBid`使用8位有符号整数，因此被其他TB依赖的TB必须能够重编号到`0..127`。这些限制不作为调度求解的硬约束，也不因超限而终止逻辑调度、XML候选生成、BDD拥塞分析或离线调优。

离线验证必须独立输出MSCCL执行兼容性结果，至少包括超限Rank、TB、channel、当前值、限制值和受影响的`transfer_id`集合。执行兼容性检查与BDD拥塞分析相互独立：即使调度超过MSCCL限制，仍须完成BDD机会分析，并按照路径、链路容量和依赖关系继续离线调优。

离线验证按以下顺序给出修改建议：

1. XML生成器首先尝试对TB重新编号，使所有被`depid`引用的TB位于`0..127`。该操作只改变XML局部标识，不得改变TB内容、依赖关系或求解调度。
2. 在不超过32个channel及TB数量限制的范围内，计算能够使每个TB不超过256个step且依赖TB可编码的最小channel数。修改channel数会改变并行开销和调度时间，因此建议只作为下一轮重新求解的输入，不得直接修改当前调度。
3. 如果当前slice大小仍导致step、TB、依赖TB或缓冲区offset超限，则枚举能够整除`total_size_bytes`的更大`slice_size_bytes`，并对每个候选值重新执行算子地址映射、channel分配和XML降低。ReduceScatter和AllToAll还须保证新的`N`能够被全局Rank数整除。建议选择满足全部执行限制的最小增量值，以保留尽可能细的传输粒度。
4. `slice_size_bytes`在当前求解、验证和调优任务中保持不变。用户采用新的slice大小后，必须开始新的求解任务并重新生成slice ID、调度和XML。

只有用户请求在线试运行或实际执行时，MSCCL兼容性才升级为强制预检查。未通过时不得加载XML或启动集合通信，避免MSCCL解析失败、越界或死锁。

每次调度均输出逻辑调度、XML和结构化验证报告。通过MSCCL兼容性检查时，XML命名为`<schedule_name>.xml`；未通过时仍输出用于离线分析的`<schedule_name>.candidate.xml`，配套报告命名为`<schedule_name>.validation.json`并设置`runtime_compatible=false`。报告必须区分语义正确性、BDD拥塞分析、性能调优结果和MSCCL执行兼容性，禁止用执行兼容性告警替代语义验证或拥塞分析。VeriCCL执行接口不得加载`runtime_compatible=false`的候选XML。

VeriCCL每次在线验证或实际执行只加载一个XML，不负责在同一communicator中注册或选择多个XML算法。MSCCL的算法注册数量及其他执行限制仅纳入离线兼容性提示，不扩展MSCCL运行时。在线验证只运行当前XML对应的`nccl-tests`基础测试，不构造包含多个集合通信算子的应用级工作负载。

### (2)算子语义与分层合成

设`c = rN + l`，其中`r`为源Rank，`l`为逻辑slice位置。六类直接求解和XML输出算子的最终语义如下：

| 算子 | 最终逻辑状态 |
| --- | --- |
| Broadcast | 所有Rank在位置`l`得到root贡献`{root*N+l}` |
| Reduce | root在位置`l`得到`{rN+l | 0<=r<P}`的AggregateState |
| AllGather | 每个Rank得到全部原始slice，输出位置为`rN+l` |
| AllReduce | 每个Rank在位置`l`得到全部Rank贡献的AggregateState |
| AllToAll | 令`q=N/P`，源Rank `r`的位置`l`发送到`floor(l/q)`，输出位置为`r*q+(l mod q)` |
| ReduceScatter | 令`q=N/P`，位置`l`的归约结果归属`floor(l/q)`，其输出位置为`l mod q` |

Scatter和Gather只作为PlanDAG内部节点，不接受为独立求解目标。AllGather表示为一组Broadcast；Reduce和Gather使用传播语义的对偶形式；ReduceScatter以及AllReduce中的ReduceScatter阶段通过求解语义正确的AllGather，再反向传播边并重建REDUCE依赖得到。反向转换后必须重新验证贡献集合、状态消费和ready time，禁止仅反转XML step顺序。归约假设满足结合律和交换律，不保证不同合法归约树产生浮点位级一致结果。

分层求解只在用户启用分层策略时使用。Planner将全局算子转换为PlanDAG，每个局部节点包含local_collective、communication_group、logical_input_map、logical_output_map、allowed_topology和shared_resources。局部通信组只用于缩小求解规模，最终必须合成为一个全局调度和一个全局XML。

通信组内Rank默认按数字升序排列，跨同构节点的Rank按对应位置一一映射。只有topo中实际存在的单向逻辑链路才能进入通信组，不生成虚拟跨节点连接。共享NIC等资源由topo定义，并参考SyCCL输入和ForestColl逻辑链路转换进入统一资源模型。局部结果仅在通信域、链路方向、容量、共享资源和语义接口完全同构时复用，不使用近似同构。

例如两台机器各4个Rank，节点内分别为`[0,1,2,3]`和`[4,5,6,7]`，且只有Rank 0和4连接NIC，则AllReduce依次执行：节点内Reduce到Rank 0和4；Rank 0与4之间ReduceScatter；Rank 0与4之间AllGather；最后执行节点内AllGather，即由网关完整结果构造的一组本地Broadcast。不得构造不存在逻辑链路的跨节点组`[1,5]`。在该分层模式下，已知严格劣于该模板的直接全局AllReduce候选可以在建模前排除。

Composer根据每个slice或AggregateState的实际ready time执行事件驱动合成，不在阶段边界加入统一barrier。每个局部阶段的输出contributors、逻辑地址和Rank必须与后继阶段输入接口完全匹配，最终按全局CollectiveSpec验证完整语义。

### (3)调度合成/求解器
不同消息规模和执行环境对延迟、吞吐及求解开销的要求不同。VeriCCL使用统一atom接口组合剪枝、构造式生成树和MILP求解，并将所有候选降低为MSCCL XML；仓库根目录的`Allgather.n16-1MB_i8_v1.xml`作为现有格式参考。求解器必须记录搜索空间是否受限、候选是否经过完整验证以及是否具有最优性证明，不能将启发式或超时incumbent表述为全局最优。

[1]硬约束
调度求解设置以下硬约束，具体实现参考TACCL、TE-CCL和SyCCL论文及代码：

a.语义守恒：不得丢失、复制或重复归约贡献，最终状态必须精确满足CollectiveSpec定义的输出集合。

b.状态使用约束：状态版本的消费、分支、SEND和REDUCE次数必须满足PayloadState及AggregateState状态转换规则。

c.拓扑与收发约束：仅允许使用topo中存在的单向链路、channel和共享资源；每次物理传输必须具有匹配的发送和接收端点。

d.因果约束：每个操作满足`st_time >= ready_time`；汇合状态的ready_time为全部输入完成时间的最大值。stage只表达语义范围，不设置全局stage barrier，允许不同slice流水执行。

e.容量约束：相同有向链路和channel的传输区间不得重叠；相反方向和不同channel可以并行，但并发channel共享该有向链路及NIC等资源的总带宽。

f.最终目标约束：输出Rank、逻辑地址、contributors及输出数量必须与算子语义完全一致；不满足最终贡献集合的可行调度不得进入目标优化。

[2]atom输入
用户可以在`atom.json`中输入`stage_num`和禁用项`(slice_id, src, dst, stage_id)`；两者均可为空。提供`stage_num`时，Planner必须生成数量一致且语义接口合法的stage；未提供时由规范化算子和分层策略推导。每个禁用项表示对应slice不得在指定stage使用该有向传输，求解器通过候选剪枝或硬约束排除它。该输入只是便于用户编写的粗粒度限制，不改变公开atom及其完整路径定义。

[3]组合求解
用户可以在`atom.json`中通过布尔字段启用或禁用求解策略。VeriCCL按照固定流水线组合已启用策略；每个策略实现为边界清晰的独立模块，并提供稳定调用接口：

a.剪枝：该模块部分通常为约束条件/前置步骤，需要结合b或c求解。可选策略包括对称性约束(参考TACCL论文和代码)，通信组与算子分层求解(参考原代码与SyCCL论文和代码，可自动/手动将算子/通信组划分成多个并行求解，最后再合并，如AllReduce可以分为"机内ReduceScatter+机间ReduceScatter+机间AllGather+机内AllGather")，最短路传输(参考TACCL，即先求解每两个节点间的最短路集合，slice只会使用该集合中的路径传输)，批量构造(参考ForestColl论文和代码，即同批次的slice使用相同传输路径，超出容量的部分划分为新的批次走其他路径)。构造式warm start对每个需求至多保留32条按链路开销、hop数和Rank顺序确定的候选简单路径，以避免全路径枚举的阶乘复杂度；该上限不删除MILP的合法CandidateEdge，未启用最短路或其他显式剪枝时，MILP仍在完整合法边集合上选择路径。批量构造以固定并发度K作为基础批容量，同批次中根、叶集合、合法路径集合和阶段接口一致的slice复用同一棵树，超过K后建立新批次并允许重新选树。

b.生成树：参考ForestColl论文和代码，为每条链路搜索合适的channel容量，并使用低开销生成树算法构造吞吐候选。

c.MILP：参考原代码与TACCL代码，将硬约束、atom输入和选择的求解约束/策略都建模成MILP，然后使用Gurobi求解器求解，需要支持局部求解(参考SyCCL代码，即可以只求解某个小通信组的Broadcast或ReduceScatter等)。每个源到叶需求使用精确流守恒，属于同一payload的需求共享单父节点树；每个选中树边还必须满足严格递增的树层级约束，禁止依赖正传输时延间接排除有向环。每个选中传输仅展开实际经过该边的成员slice atom，目标AggregateState的完整contributors作为独立树标识保留，不得将目标Rank的本地贡献误计入入站物理传输。

多种策略采用固定角色和固定流水线，用户只控制是否启用，不允许通过输入任意改变执行顺序。统一流程为：规范化CollectiveSpec、拓扑和atom输入；应用手动分层，或在未提供手动分层时执行自动分层；应用禁用atom、拓扑合法性、精确对称性和最短路径候选剪枝；通过批量构造和生成树产生可行候选及MILP warm start；使用MILP优化并保留生成树候选用于超时回退和最终比较；合成局部阶段并建立全局依赖；最后执行XML降低和完整验证。

手动分层优先于自动分层，但非法手动分层必须报错。生成树与MILP同时启用时表示“构造可行候选及warm start，再执行MILP优化”，不得解释为相互覆盖的后端；两者均禁用时属于无有效求解后端。最短路径、批量构造或其他可能排除全局最优解的策略必须在报告中设置`search_space_restricted = true`并记录限制内容。任何策略均不得静默覆盖其他策略的硬约束，约束冲突必须报告涉及的策略及具体对象。

链路性能模型以`invbw = alpha + beta`为输入一致性关系。若三者同时输入但不一致，以`invbw`为权威值，设置`beta_effective = invbw - alpha`并报告参数不一致。未校准时，并发度`K`下每个slice采用保守持续时间`D(K) = alpha + K * beta_effective`；校准后使用`b_safe(K) = min_{1<=k<=K}(B_link(k)/k)`和`D(K) = alpha + S/b_safe(K)`。channel数量通过MILP外层离散搜索，默认`K_max = 32`。相反方向可以并行，不同channel可以重叠，但共享有向链路及NIC等资源的总带宽。

[4]优化目标

输入参数`objective_mode`支持`latency`、`throughput`和`auto`，默认使用`auto`：

1. `latency`首先最小化所有最终atom的最大`ed_time`，再按字典序最小化物理通信操作数和总路径跳数。
2. `throughput`首先最小化有向链路及共享资源的最大稳态归一化负载，再最小化所有最终atom的最大`ed_time`。对资源`q`，归一化负载定义为`L_q = sum_i(D_i(K))/C_slot(q,K)`，其中分子只累计实际使用该资源的物理传输持续时间，`C_slot`为该资源在固定K模型中的可并行slot数；因此`L_q`和`max_q(L_q)`的单位均为微秒，表示资源拥塞时间，不再除以候选makespan。该定义与MILP吞吐目标、`maximum_normalized_resource_load`指标及候选排序完全一致，避免把人为延长调度误判为更低负载。
3. `auto`先求解latency候选并通过动态并发事件模拟得到当前`total_size_bytes`和`slice_size_bytes`下的验证完成时间，同时计算吞吐候选的理论下界。只有吞吐候选理论上可能显著改善当前完成时间时才继续求解throughput候选。若生成两个候选，则统一通过动态并发事件模拟，并选择验证完成时间较小的候选；完成时间相同时依次选择通信操作数更少、总路径跳数更少且稳定标识顺序更小的候选。

吞吐候选的理论下界正式命名为`throughput_time_lower_bound`，表示当前有限消息大小下throughput模式在现有保守性能模型中可能达到的最小完成时间，不表示带宽。统一使用微秒作为`alpha`、`beta`、`invbw`、atom时间、验证完成时间和该下界的内部单位；`B_link(k)`使用字节/微秒，报告可换算为GB/s；改善比例为无量纲值。

\[
throughput\_time\_lower\_bound=\max(LB_{resource},LB_{dependency})
\]

`LB_resource`通过连续流LP计算。该LP保留集合通信语义、合法拓扑、共享资源、禁用传输以及已确定的分层计划，但放宽slice不可分、整数路径、channel整数分配、TB顺序、XML限制和离散调度顺序。LP最小化时间`tau`，并为每个资源`q`设置`sum_i(f_iq) <= C_model(q) * tau`，其中`C_model(q) = max_{1<=K<=K_max}(K * b_safe(q,K))`。LP最优值为`LB_resource`。

`LB_dependency`忽略资源竞争，但保留SEND链、REDUCE汇合、阶段关系和最终贡献集合等必要因果关系；它使用当前模型允许的最快合法链路时间，取所有最终输出理论就绪时间的最大值。该下界和latency候选的动态模拟时间必须使用同一组拓扑及校准参数。连续放宽可能使下界偏松，从而触发额外throughput求解，但不得高估同一性能模型下的可实现最优时间并错误剪枝。

`auto`模式计算`gain_upper = max(0, (T_latency - throughput_time_lower_bound) / T_latency)`。输入参数`min_expected_improvement`默认值为0.01。`cv_relevant`取latency候选关键路径以及达到最大归一化负载的链路或共享资源中，所有稳定校准结果的最大变异系数。当`min_expected_improvement = 0`时筛选阈值为0，否则为`max(min_expected_improvement, 2 * cv_relevant)`。只有`gain_upper`不小于阈值时才求解throughput候选；缺少相关校准数据时使用`cv_relevant = 0`。任一相关校准结果不稳定时禁用该下界剪枝并直接求解throughput候选。

MILP达到求解时间上限时，若Gurobi已经得到可行incumbent，则提取该调度并依次执行完整集合通信语义检查、BDD机会分析、动态并发事件模拟和XML验证；必要分析与验证完成后，该候选可以参与latency/throughput候选比较。结果必须记录目标值、第一优先级目标的best bound和MIP gap、求解时间、模型总数及确定性的`model_index`，并设置`proven_optimal = false`。多目标模型不能使用普通单目标`ObjBound/MIPGap`属性，必须在第一优化pass结束时通过Gurobi多目标callback保存对应bound和gap；若该pass因超时未正常结束，则使用MIP callback保存的最后incumbent和bound计算gap。若没有可行incumbent，则在启用生成树或构造式求解器时执行回退；回退仍未得到可验证调度时，仅将该候选标记为失败，不影响其他候选。输入参数`require_proven_optimal`用于要求最终结果必须具有最优性证明。最终性能选择状态`selected_best`与最优性证明状态`proven_optimal`相互独立，禁止将“当前已验证候选中性能最好”表述为“已证明全局最优”。

求解时间同时受全局预算和单模型预算控制。`total_solve_timeout_s`默认值为10800秒，覆盖一次`solve`调用中的分层子问题、外层channel数搜索、latency/throughput候选和回退过程；`per_model_timeout_s`默认值为1800秒。每个新模型的实际时间上限为`min(per_model_timeout_s, remaining_total_solve_time)`。全局预算耗尽后不再启动新模型，但必须保留并验证已经得到的incumbent或构造式候选。全局预算按墙钟时间计算，不累计并行模型的CPU时间。

`mip_gap`默认值为`1e-4`。该值与TACCL调度模型使用的Gurobi默认值及SyCCL参数类默认值一致；TACCL路由模型使用`1e-9`，SyCCL公开示例配置使用`1e-3`，均可由用户显式覆盖。非零`mip_gap`终止只设置`within_requested_gap = true`，不设置`proven_optimal = true`。当`require_proven_optimal = true`时，实际传给Gurobi的相对gap必须为0；若在时间预算内未获得严格最优状态，则本次求解失败，而不是返回近似最优结果。

独立MILP模型可以并行求解，默认`max_parallel_models = 4`。独立模型是指其CollectiveSpec、通信域、拓扑、分层阶段接口、目标模式和channel候选均已固定，且构造当前模型不依赖其他模型的求解结果；依赖路由结果的调度模型、由latency下界筛选结果决定是否构造的throughput模型，以及同一候选的求解、验证和修复流程不得错误并行。同构通信组只求解一次并复用结果，不得将副本计为多个独立模型。

并行调度器检测可用CPU核心数`C`，当前批次实际并行数为`J = min(ready_independent_models, max_parallel_models, C)`，每个模型设置`Threads = max(1, min(12, floor(C / J)))`，保证所有并行模型的Gurobi线程总数不超过`C`。默认`solver_seed = 0`，表示使用Gurobi默认随机种子以固定内部搜索扰动；用户可以在`sketch.json`中覆盖。固定seed有利于同一模型的实验复现，但模型生成顺序、Gurobi版本、线程数、硬件或墙钟超时变化时不保证结果完全一致。并行批次仍共享`total_solve_timeout_s`墙钟预算；某个模型完成后可以启动新的独立模型，但不得动态增加仍在运行模型的线程数。

求解器使用精确签名的两级缓存。已验证缓存保存调度、目标值、best bound、MIP gap、最优性证明状态和验证报告；命中后可以跳过MILP求解，但在本次输出前仍必须重新执行集合通信语义检查、BDD机会分析、动态并发事件模拟和XML验证。warm-start缓存保存尚未验证的incumbent或中断模型的Gurobi初始解，只能作为后续求解的初始解，禁止直接输出。

缓存签名至少包含CollectiveSpec、拓扑结构、共享资源及校准参数、`total_size_bytes`、`slice_size_bytes`、禁用项、分层模板、目标模式、channel配置、`solver_seed`、模型模式版本和求解器版本。超时但验证通过的incumbent可以作为候选复用，但不能继承`proven_optimal = true`；只有模型签名和证明元数据完全一致时才能复用最优性证明。输入参数`force_resolve = true`忽略求解结果缓存，但不删除缓存文件，也不隐式忽略拓扑校准缓存。

动态并发事件模拟属于在线/离线验证模块，不直接加入MILP。现有代码中基于`self.time`的目标保留为latency模式的基础实现，链路数量和负载均衡heuristic改为明确的次级目标，不再通过未归一化加权和混合不同目标。

### (4)形式化验证器
验证与调优子系统由语义验证、BDD机会分析、动态事件模拟、XML验证、在线校准、在线算子验证和候选修复等独立模块组成。其目标是在保持集合通信语义和用户约束不变的前提下诊断候选、生成修复并选择经过验证的最佳候选，同时控制增量分析和重算开销。

[1]在线验证[可选]：在线验证分为链路校准和算子验证。链路校准使用独立基准XML测量不同channel并发度下的`B_link(k)`；算子验证通过当前XML对应的单个`nccl-tests`基础测试采集逐step时间、总算法带宽和总完成时间。Broadcast、Reduce、AllGather、AllReduce、AllToAll和ReduceScatter分别使用对应的基础性能测试；Scatter和Gather不提供独立在线测试。每次测试只加载当前XML，不测试MSCCL多算法注册或应用级算子序列。固定128 MiB链路校准不能独立估计启动开销，因此`alpha`始终保留topo输入值。

原始MSCCL v0.7.4和`nccl-tests`不直接导出XML step级时间，因此正式性能测试与step诊断必须分离。VeriCCL必须提供适配MSCCL v0.7.4的trace补丁、构建说明和trace读取器；启用在线算子验证时，该trace能力是强制要求。现有GPU `printf`式输出和TB级聚合等待时间不足以完成逐step分析，trace实现必须改用预分配的GPU定长事件缓冲区，kernel结束后统一复制到host，禁止在通信执行期间逐step调用GPU `printf`。

每个XML step至少记录`rank`、`tb_id`、`step_index`、`transfer_id`、端点类型、peer、channel、iteration、`tb_reach_time`、`dependency_done_time`、`transfer_start_time`、`transfer_end_time`和trace状态标志。`tb_reach_time`表示TB完成前一step并开始处理当前step；`dependency_done_time`表示当前step的XML依赖全部满足；`transfer_start_time`表示peer、FIFO、credit和通信primitive均已就绪并实际开始数据操作；`transfer_end_time`表示完整slice的当前端点操作完成。`cpy`记录本地复制起止时间，`nop`记录依赖满足和完成时间。

MSCCL只有执行到某个step时才检查其依赖，无法直接记录后续step本来可以更早就绪的时间。因此XML生成器必须输出sidecar，将`(rank, tb_id, step_index)`映射到`transfer_id`、atom、flow以及降低前的完整语义前驱集合。在线分析器使用全部语义前驱的实测物理完成时间重建：

\[
semantic\_ready_i=\max_{p\in predecessors(i)} physical\_end_p
\]

初始输入slice的ready time取kernel执行基准时刻，AggregateState取全部贡献完成时间的最大值。该计算不得只使用XML单依赖格式保留的一个`depid/deps`。

同一物理传输的`s`和`r/rrc`端点通过`transfer_id`配对。完成跨Rank时钟同步后，定义`physical_start = max(start_send, start_recv)`和`physical_end = max(end_send, end_recv)`。缺少任一端点、`transfer_id`无法匹配、trace缓冲区溢出或时钟同步误差超过配置上限时，该次step trace无效。运行时使用GPU全局计时器，并通过多点CPU-GPU校准及跨节点时钟同步转换到统一时间轴；报告必须记录同步误差上界，小于该误差上界的时间差不得用于确定step先后。

在线分析器逐step计算：

\[
head\_of\_line\_wait=\max(0,tb\_reach-semantic\_ready)
\]

\[
dependency\_wait=\max(0,dependency\_done-tb\_reach)
\]

\[
peer\_resource\_wait=\max(0,physical\_start-dependency\_done)
\]

\[
transfer\_duration=physical\_end-physical\_start
\]

`head_of_line_wait`用于识别后续step已经ready但被较小step阻塞的TB队首等待；`dependency_wait`表示TB已经到达当前step但显式依赖尚未完成；`peer_resource_wait`表示依赖满足后仍在等待配对端点或通信资源；`transfer_duration`用于识别实际传输变慢和channel并发带宽下降。在线分析结果必须生成逐step trace、统一时间线和瓶颈报告，并将每个瓶颈关联到`transfer_id`、atom、flow、Rank、TB、step、lane及等待类型。

正式的5次预热和20次性能测试使用未启用trace的release MSCCL；在线诊断另外执行一次相同XML、消息大小和参数的trace运行，trace结果不参与正式性能统计。启用在线算子验证时，逐step trace必须完整成功；trace失败不否定调度的离线语义正确性和MSCCL格式正确性，但必须设置`online_operator_validation = failed`，不得将该候选标记为在线验证完成，也不得根据不完整数据执行在线调优。未请求在线算子验证时不要求trace运行。

每轮在线验证默认预热5次并正式执行20次，使用正式执行时间的中位数作为调度时间，同时记录P95、均值、标准差和变异系数。变异系数超过5%时重新执行该轮，最多重测3轮；达到上限后保留全部样本并在报告中标记结果不稳定。测试必须启用`nccl-tests`正确性检查，并使用当前调度的精确消息大小、数据类型、归约操作、root和原地/非原地模式，不得使用最佳单次时间替代统计结果。

`B_link(k)`不依赖目标调度是否实际出现并发度`k`。VeriCCL预先为不同channel并发度生成基准测试XML，基准消息大小固定为128 MiB。机内链路只在`1机*2卡`环境测试，机间链路只在`2机*1卡`环境测试；其余逻辑链路根据拓扑中的精确同构关系复用对应链路类别的测量结果，不执行更大规模的链路基准测试。基准结果按机内和机间链路类别分别保存，并用于更新拓扑参数和重新求解。

基准XML使用当前任务的`slice_size_bytes = S`，基准slice数量为`N_bench = 128 MiB / S`。输入参数`max_calibration_channels`默认值为32，有效最大并发度为`K_effective = min(max_calibration_channels, 32, N_bench)`，保证每个活跃channel至少传输一个完整slice。VeriCCL为每个`k = 1..K_effective`生成独立基准XML并逐一测试，不使用插值、外推或提前停止。如果`S`不能整除128 MiB，则跳过该任务的在线链路校准并在报告中提示，不自动改变slice大小，也不使用其他slice大小的基准结果替代。

固定128 MiB消息和固定slice大小不能唯一分离实测启动开销与单slice传输开销，因此在线校准保留topo输入中的`alpha`。设并发度`k`下完整传输批次的P95耗时为`D_safe(k)`，则更新：

\[
invbw=D_{safe}(1), \qquad beta=\max(invbw-alpha, \epsilon)
\]

\[
B_{link}(k)=\frac{kS}{\max(D_{safe}(k)-alpha, \epsilon)}
\]

求解器使用`b_safe(K) = min_{1<=k<=K}(B_link(k)/k)`以及`D(K) = alpha + S/b_safe(K)`计算保守并发开销。若`D_safe(k) <= alpha`，该测量点无效，保留原拓扑参数并在报告中记录。在线校准只更新`beta`、`invbw`和`B_link(k)`，不修改`alpha`。

基准XML和测量结果均允许缓存。环境签名完全匹配时，VeriCCL默认自动复用已有测量结果，不重复运行基准测试。签名至少包含链路类别、拓扑签名、GPU和NIC型号、CUDA/NCCL/MSCCL版本、Simple协议、slice大小、128 MiB基准消息大小、channel并发度、`NCCL_BUFFSIZE`、MSCCL chunk/slice steps以及影响传输路径的关键环境变量。任一字段不匹配时缓存失效并重新测试。用户可以设置`force_recalibrate=true`忽略缓存并强制重测。

链路基准统一使用定制Broadcast XML和`broadcast_perf`。Rank 0作为root，将128 MiB数据按当前slice大小划分，并在并发度`k`下将slice分配到`k`个channel。该基准只激活root到目标Rank的单向传输，不包含REDUCE计算，也不同时激活反向链路。每类机内和机间链路只测试一个代表方向，反向链路及其他有向链路仅在拓扑证明其与代表链路精确同构时复用测量结果。

并发度`k`的基准XML按照`channel = slice_index mod k`和`wave = floor(slice_index/k)`轮转分配slice。只有恰好包含`k`个传输的完整批次参与`D_safe(k)`和`B_link(k)`计算；批次耗时为该批次最早step开始时间到最晚step结束时间。尾部不足`k`个slice的批次仍正常执行并计入`broadcast_perf`总完成时间，但不计入并发度`k`的链路样本。基准不得复制、填充或拆分slice以补齐尾部。

[2]离线BDD分析：BDD是离线拥塞与等待分析器，不承担集合通信最终语义判定，也不直接修改调度。发现可调优机会不表示当前调度无效；CollectiveSpec、AggregateState、依赖、缓冲区和XML验证器分别负责对应的正确性判定。

一个flow表示某个子通信算子中一个slice从树的一端到目标叶节点的完整链路，子通信算子通常对应一个stage。归约类stage使用实际物理传输方向记录反向flow。不同`[root, leaf]`的flow可以使用相同资源，因此BDD不得只在相同端点组内比较，而应依据实际有向链路、channel和共享资源建立全局索引。树的公共前缀和AggregateState的公共后缀通过`transfer_id`去重；两个成员slice首次形成同一AggregateState后，成员间的路径比较在汇合点截断。

链路调度资源的基本单位为`LaneKey = (src_rank, dst_rank, channel)`。MILP已经保证同一lane上的传输区间不重叠，因此flow拥塞分析不查找同一lane内的传输重叠，而是检查非叶子Rank上的状态是否满足`ready_time < st_time`。等待区间为`[ready_time, st_time)`；若存在能够承载候选flow且具有更早空闲位置的合法lane，则该等待形成可调优候选。不同channel可以并行，但同一有向链路的全部channel仍共享`B_link(k)`描述的总带宽；BDD使用LaneState查找空闲位置，动态事件模拟负责判断增加并发后是否实际降低完成时间。

BDD按照任务拆分符号关系。FlowCongestionBDD只对紧凑的`flow_id`、`candidate_flow_id`、`demand_id`和`lane_id`执行交、并、差、补等集合操作；Rank、stage、`ready_time`、`st_time`、`ed_time`、路径边和lane空闲区间保存在FlowRecord、TransferDemand和LaneState元数据中，不展开为BDD变量。TBOrderBDD独立分析同一TB内的实际顺序、必要顺序和ready time逆序，输出可交换step对；该任务才使用紧凑的`tb_id`、`op_id`和`step_index`关系，不携带完整flow路径。

FlowCongestionBDD仅输出`FlowReplacementHint`，至少包含source flow、候选替换flow集合、分歧Rank、等待transfer、瓶颈lane、等待时间和候选最早开始时间。它只执行拓扑、禁用项和基本flow兼容性预筛选，不执行链路试替换、依赖切换、后缀修复或时间重排。TBOrderBDD同样只输出顺序调优提示。所有实际修改均进入离线调优模块。

[3]离线调优/修复：调优器根据离线BDD提示、动态事件模拟和可用的在线诊断证据生成候选，并以简洁、专业的报告说明瓶颈位置、根因和修复内容。调优持续到时间预算耗尽、候选空间耗尽、达到最大迭代次数或无法找到进一步改善；最终输出`selected_best`候选，不将“无法继续改善”等同于全局最优证明。

调优器在独立的copy-on-write候选中消费BDD提示。flow替换从source flow与candidate flow的首个分歧Rank开始，只替换公共前缀之后的后缀；公共前缀和仍被其他flow引用的物理传输保持不变。替换后根据stage接口重新计算实际可达节点或contributors，修复因替换而缺失的下游交付、producer-consumer依赖、AggregateState汇合和公共后缀。修复后的最终CollectiveSpec语义、用户目标Rank或owner、禁用项和全部状态约束必须与原问题完全一致。

为降低调优开销，首先使用已有flow、transfer、依赖、lane、共享资源和AggregateState索引执行增量贪心修复，选择新增传输时间、hop、lane等待、共享资源负载和修复操作数量综合代价最小的合法后缀。贪心修复失败但BDD提示的预计收益仍为正时，仅对分歧Rank之后的受影响slice、状态、lane和资源运行局部MILP；固定公共前缀、无关flow及不受影响的lane顺序。局部MILP仍无法得到合法结果时拒绝该提示，单个BDD提示不得触发全局重新求解。

时间重计算只从影响闭包开始。初始闭包包括变更的传输、下游语义依赖、插入位置之后的同lane操作、并发度发生变化的同有向链路操作和受影响共享资源上的操作；时间变化继续向下游传播并动态扩展闭包，直到达到不动点。调度器使用新的ready time、lane可用时间、依赖完成时间和`B_link(k)`重新执行确定性的最早开始时间排列。候选先经过增量事件模拟，性能未改善时立即拒绝；改善后再执行完整语义、AggregateState、约束、BDD重新分析、动态事件模拟、BufferPlan、XML和死锁验证，全部通过后才允许替换当前最佳候选。

自动调优不得修改或写回用户输入文件，而是为每个候选建立独立`TuningOverlay`。overlay允许调整channel数量及slice分配、合法路径权重和选择、候选级临时禁用传输、batch划分、生成树根和树结构、调度顺序、MILP参数、warm start以及局部或全局重求解范围；仅在用户启用自动分层且没有手动分层时，才允许探索其他合法分层模板。

CollectiveSpec、总数据量、slice大小、slice ID、用户禁用atom、手动分层、拓扑连通性、共享资源定义以及用户明确禁用的求解策略均不可由调优器修改。拓扑性能参数只能由在线校准更新。自动添加的临时禁用传输仅作用于当前候选，可以在后续候选中撤销，禁止写回`atom.json`。无法在这些边界内继续调优时，只能在报告中给出需要用户开始新任务的输入修改建议。

每个输出XML对应的验证报告必须完整记录该候选请求启用的策略、实际应用的策略、全部策略参数、`TuningOverlay`相对规范化输入的变更、触发该变更的瓶颈和诊断证据，以及该策略是否限制了搜索空间。报告必须能够仅根据输入快照和所记录策略复现该XML的求解配置。

每轮调优根据当前瓶颈生成一个或多个新候选，并对每个候选重新执行集合通信语义检查、BDD机会分析、动态并发事件模拟和XML验证。未启用在线验证时，只有动态并发事件模拟完成时间严格小于当前最佳值的候选才被接受。启用在线验证时，使用正式测试的中位完成时间计算`improvement = (T_best - T_new) / T_best`，且只有`improvement >= max(min_tuning_improvement, 2 * max(CV_best, CV_new))`时才接受；`min_tuning_improvement`默认值为0.01。

正确性、语义、依赖、缓冲区或死锁验证失败的候选直接拒绝。BDD正常完成并发现优化机会不属于验证失败；BDD分析器自身未能完成必需分析时记录`analysis_error`，该候选不得进入最终选择。MSCCL执行不兼容的候选可以保留用于离线分析，但不得进入在线性能比较。所有候选使用精确签名去重，禁止重复求解相同候选或在调优状态之间循环。未被接受的候选及其拒绝原因仍写入验证报告，且不得覆盖当前最佳候选。

输入参数`max_tuning_iterations`默认值为20。调优在达到验证调优时间上限、候选空间耗尽、无法生成新的合法修复或达到最大迭代次数时停止。最终结果始终选择全部历史中通过相应验证且性能最好的候选，不得默认选择最后生成的候选。

`total_verification_timeout_s`默认值为10800秒，按墙钟时间覆盖一次`vericcl verify`调用中的离线验证、在线校准、在线测试、候选重求解和重新验证。每次内部重求解的有效总预算为`min(total_solve_timeout_s, remaining_verification_time)`；每个MILP的有效预算进一步限制为`min(per_model_timeout_s, remaining_solve_time, remaining_verification_time)`。达到验证总预算后不得启动新的验证或候选求解。截止时尚未完成全部必要检查的候选不得被接受，最终仍返回此前已经完整验证的最佳候选；`max_tuning_iterations`不得突破该总预算。

### (5)测试与验收

VeriCCL建立独立`tests/`测试体系。纯软件单元测试覆盖CollectiveSpec和输入校验、slice编号、六类直接算子的逻辑地址映射、AggregateState状态转换、atom路径和ready time、禁用项、分层计划、AG到RS的对偶转换、缓冲区offset、EndpointAtom配对、TB排序、NOP依赖、死锁检测、BDD汇合截断、LaneState空闲窗口、FlowReplacementHint、增量后缀与依赖修复、动态事件模拟、step trace解析与端点配对、语义ready time重建、TB队首等待分解、缓存签名和报告生成。

小规模Gurobi集成测试使用2或4个Rank及少量slice，覆盖Broadcast、Reduce、AllGather、AllReduce、AllToAll和ReduceScatter，并覆盖原地/非原地、latency/throughput/auto、禁用传输和`2x2`分层拓扑。每个求解结果必须通过完整集合通信语义、完成BDD机会分析、通过事件模拟和XML验证。

XML golden测试在规范化稳定标识后比较输出，检查buffer、offset、`s/r/rrc/cpy/nop`、`depid/deps`和单向TB，并覆盖合法XML、死锁XML及MSCCL不兼容candidate XML。属性与变形测试至少验证同构Rank重编号等价性、AG反向得到RS的语义正确性、最终贡献集合完整且无重复，以及同一有向链路和channel的区间不重叠；随机微型拓扑不得生成非法调度。

硬件测试单独标记，覆盖`1机x2卡`机内测试、`2机x1卡`机间测试、128 MiB校准及六类`nccl-tests`基础测试。没有对应硬件或运行时环境时必须报告`not_run`，不得伪装为通过，也不阻塞纯软件CI。

最终验收要求全部纯软件测试通过；新增或实质修改模块的行覆盖率不低于90%；集合通信语义、AggregateState、BDD和XML依赖等关键不变量同时具有正向与负向测试；所有源代码、测试代码、生成XML和JSON诊断均不包含中文字符。每完成一个模块立即运行该模块测试，阶段结束后运行全量回归；未执行的Gurobi或硬件测试必须在最终报告中单独列出。

## 3.工作流程

### (1)输入
输入文件固定为`topo.json`、`sketch.json`和`atom.json`。`topo.json`沿用现有拓扑示例的Rank、单向链路、channel及性能参数，并扩展SyCCL风格的共享NIC等资源；存在的链路默认允许使用，不额外输入Rank顺序，通信组内及同构节点间均按Rank数字顺序映射。迁移前格式参考旧`./taccl/examples/`，迁移后的规范示例统一放在`./vericcl/examples/`。

`sketch.json`包含唯一CollectiveSpec、`total_size_bytes`、`slice_size_bytes`、求解预算、目标模式、校准选项及其他超参数。`atom.json`包含布尔求解策略、可选手动分层、可选`stage_num`和禁用项`(slice_id, src, dst, stage_id)`；手动分层和禁用项均可为空。输入加载后必须生成不可变`resolved-input.json`，后续调优只能通过TuningOverlay派生候选，不得写回原始JSON。

### (2)输出
每次任务输出到独立目录`vericcl_<operator>_<scale>_<run_id>/`，其中包括规范化输入快照`resolved-input.json`、全局索引及总结`run-summary.json`、调度目录`schedules/`、逐调度报告目录`reports/`和可选trace目录`traces/`。

每个调度使用确定性英文文件名`vericcl_<operator>_<scale>_iter-<NNN>_selected-best-<true|false>.xml`，未通过MSCCL执行兼容性检查时使用`.candidate.xml`。每个XML或candidate XML必须具有同名`.validation.json`报告；任务结束后，最终选择的调度另外输出`vericcl_<operator>_<scale>_final.xml`及对应报告。XML与报告通过SHA-256绑定。文件名使用`selected-best`而不是`optimal`，避免将当前验证候选中的最佳结果误写为已经证明的全局最优结果。

逐XML报告至少包含规范化输入哈希、请求策略、实际策略、策略参数、`TuningOverlay`、分层计划、channel配置、缓冲区映射、solver状态、目标值、best bound、MIP gap、求解时间、各验证维度状态、在线性能统计、step trace状态、接受或拒绝原因，以及`selected_best`、`proven_optimal`和`search_space_restricted`。`run-summary.json`列出所有XML、对应报告、父候选、调优迭代关系和最终选择结果。

验证状态不得压缩为单一布尔值。输入非法或无法形成语义合法问题时标记`fatal`且不输出XML；集合通信语义、依赖、缓冲区、死锁或XML格式验证失败时标记`invalid`且不得成为最终结果；BDD发现优化机会只写入分析结果，不改变正确性状态，必需BDD分析未完成时单独记录`analysis_error`并禁止该候选进入最终选择。MSCCL执行不兼容时标记`warning`、输出candidate XML并禁止执行；在线算子验证所需trace不可用或不完整时设置`online_operator_validation=failed`，禁止声明在线验证完成及执行在线调优，但不否定离线有效且满足MSCCL兼容性检查的XML；校准不稳定时禁止相应剪枝和在线调优决策，但不否定XML执行兼容性。全部必需检查通过时标记`valid`。各JSON字段、诊断文本和生成代码均不得包含中文字符，诊断使用简洁、专业的英文。

### (3)Workflow
用户在仓库根目录安装VeriCCL：

```sh
pip install -e .
```

求解命令为：

```sh
vericcl solve \
  --topology /path/to/topo.json \
  --sketch /path/to/sketch.json \
  --atoms /path/to/atom.json
```

集合通信语义默认从`sketch.json`中的CollectiveSpec读取，不要求重复传入`--operator`。验证现有XML使用：

```sh
vericcl verify \
  --topology /path/to/topo.json \
  --sketch /path/to/sketch.json \
  --atoms /path/to/atom.json \
  --xml /path/to/schedule.xml \
  [--online] \
  [--tune] \
  [--timeout-s 10800]
```

未指定`--tune`时只验证当前XML；指定后按照`max_tuning_iterations`和总时间预算迭代调优。`--online`启用链路校准、基础算子性能测试和强制逐step trace算子验证。调优停止表示在当前候选空间和预算内未找到进一步改善，不表示已证明全局最优。
