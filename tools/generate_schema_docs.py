"""从实际 schema 生成字段清单与 JSON Schema，防止教程字段漂移。"""
import argparse
import json
from pathlib import Path

from vericcl.input.schema import SCHEMAS, public_schemas

HELP = {
    'name': '拓扑名称，仅用于识别。', 'ranks': '参与任务的逻辑 rank 数；节点内列表则列出其 rank。',
    'nodes': '节点分组，完整覆盖 rank；节点编号不代替实际主机地址。', 'id': '本配置内唯一标识。',
    'gateways': '可作为该节点对外连接端点的 rank。', 'directed_links': '有向直连或交换机抽象链路；反方向须单独提供。',
    'src': '发送端 rank。', 'dst': '接收端 rank。', 'alpha_us': '每次服务固定开销，单位微秒。',
    'beta_us_per_byte': '每字节服务开销，单位微秒/字节，带宽为其倒数。',
    'max_channels': '允许的 channel 数上限；任务默认搜索 1 到 7，显式上限可到 16。',
    'resources': '该链路占用的共享资源标识。', 'physical_path': '抽象链路对应的物理路径说明。',
    'shared_resources': '声明不同抽象链路的共同瓶颈，防止重复计算物理带宽。',
    'capacity_bytes_per_us': '资源总有效带宽，单位字节/微秒。', 'links': '属于资源的有向链路 [src,dst]。',
    'kind': '共享资源类别。', 'groups': '用户提供的候选分层组。', 'local_costs': '当前数据类型和归约操作下的本地开销初值。',
    'rank': '逻辑 rank 编号，从 0 开始。', 'copy': '本地读取、写入或复制的初始代价。', 'reduce': '本地归约的初始代价。',
    'collective': '对应用可见的算子语义。', 'operator': '集合通信算子；手工内部节点还可使用 Scatter/Gather。',
    'datatype': '数据类型；默认 float32，实际支持需通过运行时检查。',
    'root': 'Broadcast/Reduce 的外部根，默认 0；手工节点表示该节点的根。',
    'reduction_op': 'Reduce/AllReduce/ReduceScatter 的归约操作，默认 sum。其他外部算子须省略或为 null。',
    'inplace': '是否使用合法的原位缓冲关系，默认关闭。', 'hyperparameters': '应用输入规模与最小 slice。',
    'total_size_bytes': '每 rank 输入字节数 M；Broadcast 指 root 消息大小。',
    'slice_size_bytes': '固定最小 slice 字节数 S；省略时先搜索合法分区。N 由 M/S 派生。',
    'reference_slice_size_bytes': '解释原始 slice_id 硬约束的参考分区；重分区时仍绑定原字节范围。',
    'solver': '求解器、CPU 配额和求解策略。', 'backend': '仅支持 SCIP；缺失时直接报错。',
    'per_model_timeout_s': '单个求解模型的时间上限，仍受父阶段剩余额度约束。', 'cpu_budget': '整个任务可使用的 CPU 配额。',
    'max_threads_per_model': '单模型线程上限，默认 12；实际还受 CPU 配额限制。',
    'max_parallel_models': '同时运行模型数的上限；省略时由预算管理器控制。',
    'solver_seed': 'SCIP 随机种子。', 'mip_gap': '允许的数值 MIP gap；不构成完整修复空间证明。',
    'require_proven_optimal': '是否要求当前求解模型达到数值最优状态。', 'bdd_backend': '分析使用的 BDD 后端；CUDD 默认，autoref 仅显式选择。',
    'budgets': '各阶段时间上限和最大调优轮数；时间均为秒。',
    'max_initial_synthesis_s': '本次初始合成总时间上限。', 'max_online_validation_per_round_s': '一轮在线校准、trace 和实测的累计上限。',
    'max_online_validation_total_s': '整个任务在线验证累计上限。', 'max_offline_validation_per_round_s': '一轮离线检查与分析累计上限。',
    'max_offline_validation_total_s': '整个任务离线检查与分析累计上限。', 'max_repair_per_round_s': '一轮所有局部/结构/分区修复累计上限。',
    'max_repair_total_s': '整个任务修复累计上限。', 'max_run_wall_s': '可选的整个任务墙钟上限。',
    'cleanup_reserve_s': '从剩余时间中保留的收尾时间；不会延长父期限。', 'max_tuning_rounds': '最多开始多少轮调优；0 表示建立初始结果后结束。',
    'tolerances': '浮点数值比较容差，执行时须明确提供。', 'absolute': '绝对误差容差。', 'relative': '相对误差容差。',
    'calibration': '独立原语测量与代价模型校准。', 'enabled': '启用本配置段的功能。',
    'launches': '每轮独立进程启动数；同次启动内的迭代不视为独立启动。',
    'warmup_iterations': '每次启动的预热次数；详细 trace 在预热后清零再记录一次。',
    'timed_iterations': '无详细 trace 时每次启动的计时次数。', 'max_rounds': '稳定性不满足时最多进行的测量轮数。',
    'max_cv': '同轮样本变异系数上限。', 'max_relative_fit_error': '代价拟合允许的最大相对误差。',
    'job_timeout_s': '完整单次测量任务上限，含启动、重试、收集与清理。', 'payload_sizes': '额外探针大小；不改变应用的 M/S。',
    'trace_and_measure': '同一开关控制详细 trace 和候选正式实测收益门槛，默认关闭。',
    'pairs': '预先确定的 AB/BA 成对启动数，须为至少 2 的偶数。', 'min_relative_gain': '实测要求的最低相对改善。',
    'task_alpha': '整个任务允许的统计错误概率预算，最大 0.05。',
    'single_attempt_alpha_cap': '单次比较的错误概率上限，仍受任务账本分配约束。',
    'seed': '预先确定采样顺序的种子。', 'trace_capacity_bytes': '每 rank 的详细 trace 缓冲区字节数，至少容纳一次完整记录。',
    'repair': '修复阶段的分析/求解分工。', 'mode': 'strategy_first 先选策略，joint 联合选择合法决策。',
    'candidate_batch_size': '每批进入局部 SCIP 的候选数；不表示总搜索上限。', 'analysis_batch_size': '每批分析建议数；耗尽后继续扩展。',
    'runtime_compatibility': '可关闭的完整静态运行时兼容检查；基础启动安全检查始终保留。',
    'history': '跨任务调度与绝对校准缓存，两个开关互相独立，默认关闭。',
    'schedule_reuse': '检索同构且相对带宽接近的历史调度，再适配并重新验证。',
    'calibration_reuse': '仅复用精确设备/运行时/类型/模式身份下仍有效的测量组件。',
    'relative_bandwidth_tolerance': '映射后的各带宽缩放比例最大值/最小值减一的允许上限。',
    'max_schedule_candidates': '快速历史检索最多检查的候选数。', 'calibration_max_age_s': '校准缓存的最大年龄。',
    'schedule_index': '历史调度索引路径；相对路径从案例目录解析。', 'calibration_index': '校准索引路径；相对路径从案例目录解析。',
    'live': '辅助常驻在线模式，主离线闭环不依赖其开启。',
    'request_prepare_timeout_s': '请求准备和原输入保护期限。', 'execution_attempt_timeout_s': '每次 GPU 尝试期限。',
    'background_tuning_task_timeout_s': '一次后台调优任务期限。', 'version_prepare_timeout_s': '新版本准备期限。',
    'control_phase_timeout_s': '一个控制阶段的期限。', 'recovery_timeout_s': '整个失败恢复过程的固定总期限。',
    'close_cleanup_timeout_s': '关闭、后台结束和资源回收的共同期限。',
    'window_requests': '一个合格遥测窗口的请求数。', 'consecutive_windows': '触发前要求连续同方向变化的窗口数。',
    'relative_change_threshold': '相对基线的变化阈值。', 'baseline_max_cv': '建立基线所需稳定性上限。',
    'min_residence_requests': '采用版本后，普通更新前至少保留的成功请求数。',
    'max_retries': '失败后允许的额外完整请求重试次数；默认 2。', 'external_telemetry_path': '调用者维护的带内遥测 JSONL 文件路径。',
    'external_max_age_s': '接受外部遥测的最大年龄。', 'fault_injection': '仅显式启用的开发故障注入。',
    'request_id': '需要注入失败的逻辑请求号；0 关闭。', 'attempts': '该请求前多少次尝试在实际完成后被标记失败。',
    'synthesis_method': 'auto 按 M_switch 选择；小消息 hierarchy，大消息 constructive_trees。',
    'M_switch': 'auto 模式必须显式给出的消息大小分界，单位字节。', 'stage_num': '逻辑阶段数硬约束；省略时自动确定。',
    'forbidden_transfers': '[原 slice_id,src,dst,stage]，只禁止该阶段的这条直连边，仍可合法绕路。',
    'strategies': '可选初始求解策略，默认不启用。', 'symmetry': '启用对称约简。', 'shortest_paths': '启用候选路径缩减策略。',
    'batching': '显式启用分层求解的批处理开关；批量生成树方法按自身规则管理 batch。',
    'objective_mode': '显式选择延迟/吞吐目标，或由消息规模自动选择。',
    'manual': '显式手工计划，仅在 enabled=true 时使用。', 'stage_id': '逻辑阶段编号，从 0 开始。',
    'group': '该手工节点允许使用的 rank 集合。', 'relay_ranks': '组内允许中转的 rank；不能引入组外 rank。',
    'depends_on': '该手工节点依赖的节点 ID。', 'input_states': '手工节点的输入状态合同。', 'output_states': '手工节点的输出状态合同。',
    'offset': '逻辑数据位置编号。', 'contributors': '该状态包含的原始 slice_id 集合。',
    'rank_map': '每个 rank 的实际主机和设备位置，必须完整且无重复设备。', 'host': 'SSH/MPI 均可访问的主机名或地址。',
    'device': '该 host 进程可见的 CUDA 设备编号。', 'environment': '运行时环境变量白名单，值均为字符串。',
    'paths': '显式工具链和可执行文件路径。', 'cuda_home': '已有 CUDA 安装根目录。', 'mpi_bin': '含 mpirun、mpicxx 的目录。',
    'msccl_root': '编译后的 MSCCL 根目录，含 include/lib。', 'nccl_tests_root': '可选 nccl-tests 根目录。',
    'vericcl_driver': '独立原语测量程序 vericcl-probe 的绝对路径。', 'vericcl_native': '完整调度及常驻程序 vericcl-native 的绝对路径。',
    'device_memory_budget_bytes': '设备总预算；0 使用启动时可用显存界，仍检查实际分配。',
    'protocol': '当前固定使用 Simple。', 'scratch_bytes_per_rank': '每 rank 允许的 scratch 字节数。',
    'limits': '运行时结构上限，可设置更小值，不能超过编译支持值。', 'steps_per_tb': '每 TB 最大指令数。',
    'tbs_per_rank': '每 rank 最大 TB 数；还须满足实际设备驻留能力。',
    'send_tbs_per_channel': '每 channel 发送 TB 上限。', 'recv_tbs_per_channel': '每 channel 接收 TB 上限。',
    'dependency_id': '依赖编号上限。', 'buffer_offset': '缓冲区 slice 偏移上限。', 'tb_layout': '默认按发送/接收分开编排；fused 须显式选择。',
    'control': '常驻控制面配置。', 'bind_address': 'rank 0 的可达监听地址。', 'port': '控制 TCP 端口。',
    'unix_socket': '控制器所在节点的 Unix socket 绝对路径。',
}
ENV = {
    'PATH': '子进程工具搜索路径。', 'LD_LIBRARY_PATH': '动态库搜索路径，优先目标运行时与 MPI。', 'PYTHONPATH': 'Python 模块搜索路径。',
    'CUDA_HOME': 'CUDA 根目录。', 'CUDA_PATH': 'CUDA 根目录兼容变量。', 'CUDA_VISIBLE_DEVICES': '设备可见范围，会改变 device 编号含义。',
    'NCCL_ALGO': '保留的 NCCL 算法接口。', 'NCCL_PROTO': '运行协议，任务使用 Simple。',
    'NCCL_BUFFSIZE': '每 channel 通信缓冲配置，须是 1 到 2147483647 的十进制字节字符串。',
    'NCCL_DEBUG': '运行时日志级别。', 'NCCL_DEBUG_SUBSYS': '运行时日志子系统。', 'NCCL_IB_HCA': '明确使用的 IB/RoCE 设备。',
    'NCCL_IB_GID_INDEX': '显式 GID 编号。', 'NCCL_SOCKET_IFNAME': '通信网卡选择。', 'NCCL_P2P_DISABLE': '是否禁用 GPU P2P 路径。',
    'NCCL_SHM_DISABLE': '是否禁用主机共享内存路径。', 'MSCCL_XML_FILES': '既有 MSCCL XML 加载接口；原生版本接口显式绑定计划。',
    'NCCL_TESTS_ROOT': '可选 nccl-tests 根目录。', 'MSCCL_ROOT': '运行时根目录。', 'OMP_NUM_THREADS': 'OpenMP 线程数。',
    'OMPI_MCA_btl': 'Open MPI BTL 组件。', 'OMPI_MCA_pml': 'Open MPI PML 组件。', 'UCX_NET_DEVICES': 'UCX 网络设备。', 'UCX_TLS': 'UCX 传输选择。',
    'OMPI_MCA_btl_tcp_if_include': 'Open MPI TCP 数据通道使用的接口或网段；各节点须互相可达。',
    'OMPI_MCA_oob_tcp_if_include': 'Open MPI 控制通道使用的接口或网段；可与 TCP 数据通道显式使用同一网段。',
}


def rows(spec, prefix):
    for key, child in spec.get('properties', {}).items():
        path = prefix + '.' + key
        description = ENV[key] if prefix == 'runtime.environment' else HELP[key]
        if path == 'atom.manual.nodes':
            description = '手工逻辑节点；每个节点给出组、算子、依赖及可选状态合同。'
        default = json.dumps(child['default'], ensure_ascii=False) if 'default' in child else '未指定'
        if path == 'sketch.collective.root': default = '适用时 0'
        if path == 'sketch.collective.reduction_op': default = '适用时 sum'
        limits = []
        for name, symbol in [('minimum', '≥'), ('exclusiveMinimum', '>'), ('maximum', '≤')]:
            if name in child: limits.append(symbol + str(child[name]))
        if 'enum' in child: limits.append(' / '.join(map(str, child['enum'])))
        required = '必填' if key in spec.get('required', []) else '按启用条件检查' if 'default' not in child else '可省略或 null'
        yield f'| `{path}` | {child["type"]} | `{default}` | {required}；{"，".join(limits) or "—"} | {description} |'
        yield from rows(child, path)
        if child['type'] == 'array': yield from rows(child['items'], path + '[]')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args(argv)
    directory = args.project_root / 'docs/input-schemas'; directory.mkdir(parents=True, exist_ok=True)
    for name, schema in public_schemas().items():
        (directory / (name + '.json')).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + '\n')
    intro = '''# 输入字段参考

此表由 `vericcl/input/schema.py` 生成。四份 JSON 都拒绝重复键、未知字段和非有限数；原文件不会被覆盖。省略或 null 表示未指定：有默认值时补入默认，无默认但本次启用功能需要的字段报错。字段在新旧位置重复出现时，即使值相同也报错。

M/S 必须整除，S 必须包含完整的数据类型元素；AllToAll/ReduceScatter 的 N=M/S 还须能被 rank 数整除。禁止直接输入 N 或 input_chunkup。不适用算子的 root/reduction_op 可以省略或 null，不能提供实际值。

命令行 `--set 路径=JSON` 可以补入未指定值。覆盖已有不同值须加 `--override-input`。原文件保存在 `input_snapshot`，最终有效配置和来源分别写入 `effective-request.json`、`value-sources.json`。配置字段不读取 shell 占位符，请在写 JSON 时使用实际绝对路径。

solve/live 必须给初始合成预算和最大轮数；轮数大于 0 时还须给修复预算。所有处理入口需要单模型/CPU、离线每轮/累计及清理预算。校准启用或使用 calibrate 时，需要在线预算、拟合误差和单任务期限；trace/实测启用时还需要 trace 容量和任务期限。历史复用各自要求索引及对应容差/年龄。live 要求全部七个阶段期限和控制端点。运行时数值检查使用显式绝对/相对容差。

JSON Schema 描述标准结构；跨字段算术、启用条件、重复键和明确的旧写法迁移由 load_request 检查。算子名称忽略大小写；兼容别名为 all_gather、all_reduce、all_to_all、reducescatter。旧 hyperparameters 中明确列出的 solver 参数可迁移；atom 顶层的 symmetry、shortest_paths、batching、objective_mode 可迁移。旧 total_solve_timeout_s 仅迁移为初始合成预算。force_resolve、force_recalibrate 及旧算法布尔开关已删除。

下面的 object/array 行描述所属配置段，后续行描述其成员。array 中的 [] 表示每一个元素。更多运行方式见 [README](../README.md)，证据范围见 [适用边界](supported-configuration.md)。
'''
    parts = [intro]
    for name, spec in SCHEMAS.items():
        parts += [f'\n## {name}.json\n\n[机器可读 Schema](input-schemas/{name}.json)\n',
                  '| 字段 | 类型 | 默认值 | 要求与取值 | 作用 |\n|---|---|---|---|---|', *rows(spec, name)]
    (args.project_root / 'docs/input-schema.md').write_text('\n'.join(parts) + '\n')
    print('已生成四份 Schema 和完整字段参考。')


if __name__ == '__main__':
    main()
