# MSCCL 测量与 trace 运行时

运行时使用仓库内固定源码及九个补丁。补丁依次提供逐步 trace、主机/设备 step 一致性、轻量测量、单次 trace 控制、显式版本、准备阶段准入、六算子分发、有界中止和重复 trace 周期。日常构建使用根目录 README 中的 `runtime/build_runtime.py`，源码、补丁和最终文件均经过散列校验。

需要从官方源码复现时，在项目根目录执行以下命令；CUDA 路径与 GPU 架构按实际环境填写：

<!-- vericcl-msccl-strategy: strategy-a -->
```bash
export VERICCL_ROOT="$(pwd -P)"
export MSCCL_SRC="${TMPDIR:-$VERICCL_ROOT/examples/runtime-build/output}/vericcl-msccl-base"
git clone https://github.com/microsoft/msccl.git "$MSCCL_SRC"
git -C "$MSCCL_SRC" checkout --detach b23e9cd5dd63f82ee1c5aae7e0a2042079be903a
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" --source-root "$MSCCL_SRC" --base-tree
cp "$VERICCL_ROOT/runtime/msccl-trace/include/vericcl_trace_format.h" "$MSCCL_SRC/src/include/"
cp "$VERICCL_ROOT/runtime/include/vericcl_probe.h" "$MSCCL_SRC/src/include/"
cp "$VERICCL_ROOT/runtime/include/vericcl_plan.h" "$MSCCL_SRC/src/include/"
for VERICCL_PATCH in "$VERICCL_ROOT"/runtime/msccl-trace/patches/*.patch; do
  patch --directory="$MSCCL_SRC" --strip=1 --input="$VERICCL_PATCH"
done
make -C "$MSCCL_SRC" -j8 CUDA_HOME=/usr/local/cuda-12.8 \
  NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70" src.build
```

## 采集完整的一次调用

轻量代价测量与详细 trace 使用不同模式。详细 trace 默认关闭；正式成对性能测量始终关闭详细 trace。

创建 communicator 前设置：

- `VERICCL_TRACE_ENABLE=1`：分配记录缓冲区。省略或设为 `0` 时关闭。
- `VERICCL_TRACE_RECORDS`：每个 rank 的记录容量。单条记录为 64 字节，文件头为 40 字节。根据该 rank 实际执行的 XML 步骤数预估容量，NOP 不计为记录。
- `VERICCL_TRACE_FILE_PREFIX`：完整输出路径前缀。rank `r` 写入 `<prefix>.rank-<r>.bin`。
- `VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4` 和 `VERICCL_EXPECTED_MSCCL_SLICESTEPS=4`：创建 communicator 时核对固定 step 配置。

调用方包含 `runtime/include/vericcl_trace_control.h`，保持 communicator 对应的 CUDA device 为当前设备，且不并发提交新工作。按以下顺序执行：

1. 创建 communicator 后调用 `vericclTracePause(comm)`，再进行预热。
2. 在目标调用之前取得时钟参考样本，调用 `vericclTraceReset(comm, expected_records)`。该函数等待 GPU 完成，检查容量，并清零计数和溢出标记。
3. 只执行一次目标集合通信，等待完成。
4. 调用 `vericclTraceFinish(comm)`，保存记录并关闭后续记录；在目标调用之后再取时钟参考样本。
5. 检查所有 rank 的数据正确性、GPU 完成及清理回执，再解析文件。原始 `iteration` 保存 MSCCL 的 `workIndex`，用来核对这一次调用的身份。

`runtime/probes/main.cu --capture-trace 1 --timed 1` 实现上述步骤，用于独立链路测量程序的开发检查。它额外保存 CUDA event 测得的本设备完整调用时间。该时间包含 trace 开销；正式收益验收使用另一次关闭 trace 的运行。旧 `runner.py` 中的多次诊断采集保留兼容用途，v5 单次接口为 `capture_once`。

文件溢出或缺步骤时，完整性检查失败。解析器保留能够逐条验证的其他记录，并标明缺失位置；这些片段可支持局部分析。完整调度的瓶颈结论使用覆盖完整且因果关系可判定的记录。

## 时钟与总完成时间

v5 单次采集使用 SM 7.0 及以上设备的纳秒 `globaltimer`。采集程序在目标调用前后读取 GPU 时间，并用主机时间区间和 MPI 往返测量映射到 rank 0 的单调时钟。它只测量时钟关系，不调整系统时钟。

解析器要求参考样本包围目标调用，并保存映射误差。单个设备内的步骤时长直接由本设备时间差计算。跨设备先后关系只有在误差区间不重叠时才确定，否则标记为不确定。完整 trace 的首个步骤到最后一个步骤给出带误差的全局时间区间；每个 rank 的 CUDA event 时间另行记录，保留两种时间的含义。

`runtime/msccl-trace/tools/vericcl_clock_sync.cu` 是早期独立参考工具。v5 使用 `runtime/probes/clock_reference.cuh` 中的前后参考测量，避免用目标区间之外的一组旧样本直接外推。
