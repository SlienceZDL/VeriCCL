#include "timing.cuh"
#include <mpi.h>
#include "vericcl_trace_control.h"
#include "clock_reference.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

struct Flow { std::string kind; int src, dst, channel; };
struct Options {
  std::string job, xml, output, datatype = "float32", reduction = "sum";
  size_t payload = 0;
  int warmup = 5, timed = 20;
  bool capture_trace = false;
  size_t trace_records = 0;
  double atol = 1e-5, rtol = 1e-5;
  std::vector<int> devices;
  std::vector<Flow> flows;
};

std::string quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
std::vector<std::string> split(std::string text, char separator) {
  std::vector<std::string> values;
  std::stringstream stream(text); std::string item;
  while (std::getline(stream, item, separator)) values.push_back(item);
  return values;
}
long long integer(const std::string& text) {
  size_t end = 0;
  auto value = std::stoll(text, &end);
  if (end != text.size() || value < 0) throw std::runtime_error("invalid nonnegative integer");
  return value;
}
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string key(argv[i]);
    if (key == "probe" && i == 1) continue;
    if (i + 1 >= argc) throw std::runtime_error("missing option value: " + key);
    std::string value(argv[++i]);
    if (key == "--job-id") o.job = value;
    else if (key == "--xml") o.xml = value;
    else if (key == "--output-dir") o.output = value;
    else if (key == "--datatype") o.datatype = value;
    else if (key == "--reduction-op") o.reduction = value;
    else if (key == "--payload-bytes") o.payload = integer(value);
    else if (key == "--warmup") o.warmup = integer(value);
    else if (key == "--capture-trace") o.capture_trace = integer(value) == 1;
    else if (key == "--trace-records") o.trace_records = integer(value);
    else if (key == "--timed") o.timed = integer(value);
    else if (key == "--atol") o.atol = std::stod(value);
    else if (key == "--rtol") o.rtol = std::stod(value);
    else if (key == "--devices") for (auto& v : split(value, ',')) o.devices.push_back(integer(v));
    else if (key == "--flow") {
      auto fields = split(value, ':');
      if (fields.size() != 4) throw std::runtime_error("invalid flow");
      o.flows.push_back({fields[0], int(integer(fields[1])), int(integer(fields[2])), int(integer(fields[3]))});
    } else throw std::runtime_error("unknown option: " + key);
  }
  if (o.job.empty() || o.xml.empty() || o.output.empty() || !o.payload || o.flows.empty() ||
      o.devices.empty() || o.timed < 1 || o.atol < 0 || o.rtol < 0 || !std::isfinite(o.atol) || !std::isfinite(o.rtol))
    throw std::runtime_error("incomplete probe configuration");
  for (const auto& f : o.flows) {
    if (f.src >= int(o.devices.size()) || f.dst >= int(o.devices.size()) || f.channel >= 16 ||
        (f.kind != "send" && f.kind != "rrc" && f.kind != "copy" && f.kind != "reduce") ||
        ((f.src == f.dst) != (f.kind == "copy" || f.kind == "reduce")))
      throw std::runtime_error("invalid flow endpoints or primitive");
  }
  if (o.capture_trace && o.timed != 1) throw std::runtime_error("once-only trace requires timed=1");
  return o;
}
ncclDataType_t datatype(const std::string& name) {
  const std::map<std::string, ncclDataType_t> names = {{"int8", ncclInt8}, {"uint8", ncclUint8},
    {"int32", ncclInt32}, {"uint32", ncclUint32}, {"int64", ncclInt64}, {"uint64", ncclUint64},
    {"float16", ncclFloat16}, {"float32", ncclFloat32}, {"float64", ncclFloat64}, {"bfloat16", ncclBfloat16}};
  auto found = names.find(name);
  if (found == names.end()) throw std::runtime_error("unsupported datatype");
  return found->second;
}
ncclRedOp_t reduction(const std::string& name) {
  const std::map<std::string, ncclRedOp_t> names = {{"sum", ncclSum}, {"prod", ncclProd}, {"min", ncclMin}, {"max", ncclMax}};
  // The pinned CustomCollective dispatch aliases PreMulSum/SumPostDiv to Sum.
  // Publishing these values as an average would silently measure another op.
  auto found = names.find(name);
  if (found == names.end()) throw std::runtime_error("this runtime CustomCollective does not support reduction: " + name);
  return found->second;
}
double value(int rank, size_t slot) { return 1 + (rank * 3 + slot) % 5; }
double accumulator(int rank, size_t slot) { return 2 + (rank + slot) % 3; }
double combine(double a, double b, ncclRedOp_t op) {
  if (op == ncclSum) return a + b;
  if (op == ncclProd) return a * b;
  if (op == ncclMin) return std::min(a, b);
  if (op == ncclMax) return std::max(a, b);
  throw std::runtime_error("unsupported reduction");
}
void save(const std::string& path, const std::string& text) {
  std::string temporary = path + ".partial";
  int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) throw std::runtime_error("cannot create native evidence");
  size_t offset = 0;
  while (offset < text.size()) {
    ssize_t amount = write(fd, text.data() + offset, text.size() - offset);
    if (amount <= 0) { close(fd); throw std::runtime_error("cannot write native evidence"); }
    offset += amount;
  }
  if (fsync(fd) != 0) { close(fd); throw std::runtime_error("cannot flush native evidence"); }
  close(fd);
  if (rename(temporary.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot publish native evidence");
}

int main(int argc, char** argv) {
  int rank = -1, ranks = 0;
  Options o;
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t timing_start = nullptr, timing_end = nullptr;
  void *input = nullptr, *output = nullptr;
  VericclProbeCounter* table = nullptr;
  bool initialized = false;
  try {
    o = parse(argc, argv);
    auto type = datatype(o.datatype);
    auto op = reduction(o.reduction);
    if (o.payload % datatype_bytes(type)) throw std::runtime_error("payload is not a whole datatype element");
    if (setenv("MSCCL_XML_FILES", o.xml.c_str(), 1) != 0) throw std::runtime_error("cannot set XML path");
    MPI_Init(&argc, &argv); initialized = true;
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != int(o.devices.size())) throw std::runtime_error("MPI rank map size mismatch");
    VC_CUDA(cudaSetDevice(o.devices[rank]));
    cudaDeviceProp properties;
    VC_CUDA(cudaGetDeviceProperties(&properties, o.devices[rank]));
    if (o.capture_trace && properties.major < 7) throw std::runtime_error("trace clock requires compute capability >= 7.0");
    int blocks = 0;
    for (const auto& f : o.flows) if (f.src == rank || f.dst == rank) blocks++;
    if (!blocks || blocks > 216) throw std::runtime_error("invalid active-rank TB count");
    if (o.capture_trace) {
      size_t capacity = o.trace_records ? o.trace_records : size_t(blocks);
      if (capacity < size_t(blocks)) throw std::runtime_error("trace capacity is below the target invocation");
      if (setenv("VERICCL_TRACE_ENABLE", "1", 1) ||
          setenv("VERICCL_TRACE_RECORDS", std::to_string(capacity).c_str(), 1) ||
          setenv("VERICCL_TRACE_FILE_PREFIX", (o.output + "/trace").c_str(), 1))
        throw std::runtime_error("cannot configure trace storage");
    } else if (setenv("VERICCL_TRACE_ENABLE", "0", 1)) throw std::runtime_error("cannot disable detailed trace");
    ncclUniqueId id;
    if (rank == 0) VC_NCCL(ncclGetUniqueId(&id));
    if (MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("MPI unique-id broadcast failed");
    VC_NCCL(ncclCommInitRank(&comm, ranks, id, rank));
    if (o.capture_trace) VC_NCCL(vericclTracePause(comm));
    VC_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t elements = o.payload / datatype_bytes(type);
    size_t bytes = o.payload * o.flows.size();
    if (bytes / o.payload != o.flows.size()) throw std::runtime_error("buffer size overflow");
    VC_CUDA(cudaMalloc(&input, bytes)); VC_CUDA(cudaMalloc(&output, bytes));
    size_t entries = size_t(blocks) * 256;
    if (!o.capture_trace) {
      VC_CUDA(cudaMalloc(&table, entries * sizeof(*table)));
      VC_CUDA(cudaMemsetAsync(table, 0, entries * sizeof(*table), stream));
    }
    for (size_t slot = 0; slot < o.flows.size(); slot++)
      fill_values((char*)input + slot * o.payload, elements, type, value(rank, slot), stream);
    VC_CUDA(cudaStreamSynchronize(stream));
    if (!o.capture_trace) VC_NCCL(vericclProbeAttach(comm, table, entries));
    double target_event_us = 0;
    if (o.capture_trace) {
      VC_CUDA(cudaEventCreate(&timing_start));
      VC_CUDA(cudaEventCreate(&timing_end));
    }
    auto run = [&]() {
      for (size_t slot = 0; slot < o.flows.size(); slot++)
        fill_values((char*)output + slot * o.payload, elements, type, accumulator(rank, slot), stream);
      VC_CUDA(cudaStreamSynchronize(stream));
      if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("MPI probe barrier failed");
      if (o.capture_trace) VC_CUDA(cudaEventRecord(timing_start, stream));
      VC_NCCL(ncclCustomCollective(input, output, elements * o.flows.size(), type, op, 0, comm, stream));
      if (o.capture_trace) VC_CUDA(cudaEventRecord(timing_end, stream));
      VC_CUDA(cudaStreamSynchronize(stream));
      if (o.capture_trace) {
        float elapsed_ms = 0;
        VC_CUDA(cudaEventElapsedTime(&elapsed_ms, timing_start, timing_end));
        target_event_us = elapsed_ms * 1000.0;
      }
    };
    for (int i = 0; i < o.warmup; i++) run();
    std::string clock_samples;
    if (o.capture_trace) {
      clock_samples = vericclClockSamples(rank, ranks, stream);
      VC_NCCL(vericclTraceReset(comm, blocks));
    } else {
      VC_CUDA(cudaMemsetAsync(table, 0, entries * sizeof(*table), stream));
      VC_CUDA(cudaStreamSynchronize(stream));
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < o.timed; i++) run();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (o.capture_trace) {
      VC_NCCL(vericclTraceFinish(comm));
      clock_samples += vericclClockSamples(rank, ranks, stream);
    }
    std::vector<VericclProbeCounter> counters(entries);
    if (!o.capture_trace) VC_CUDA(cudaMemcpy(counters.data(), table, entries * sizeof(*table), cudaMemcpyDeviceToHost));
    unsigned long long incorrect = 0;
    for (size_t slot = 0; slot < o.flows.size(); slot++) {
      const auto& f = o.flows[slot];
      if (f.dst != rank) continue;
      double expected = value(f.src, slot);
      if (f.kind == "rrc") expected = combine(expected, value(rank, slot), op);
      if (f.kind == "reduce") expected = combine(expected, accumulator(rank, slot), op);
      incorrect += check_values((char*)output + slot * o.payload, elements, type, expected, o.atol, o.rtol, stream);
    }
    // Verify read-only input independently, including sender-only ranks.
    for (size_t slot = 0; slot < o.flows.size(); slot++)
      incorrect += check_values((char*)input + slot * o.payload, elements, type, value(rank, slot), o.atol, o.rtol, stream);
    unsigned long long all_incorrect = 0;
    if (MPI_Allreduce(&incorrect, &all_incorrect, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
      throw std::runtime_error("MPI correctness agreement failed");
    if (!o.capture_trace) {
      VC_NCCL(vericclProbeAttach(comm, nullptr, 0));
      VC_CUDA(cudaFree(table)); table = nullptr;
    }
    if (timing_start) { VC_CUDA(cudaEventDestroy(timing_start)); timing_start = nullptr; }
    if (timing_end) { VC_CUDA(cudaEventDestroy(timing_end)); timing_end = nullptr; }
    VC_CUDA(cudaFree(input)); input = nullptr;
    VC_CUDA(cudaFree(output)); output = nullptr;
    VC_CUDA(cudaStreamDestroy(stream)); stream = nullptr;
    VC_NCCL(ncclCommDestroy(comm)); comm = nullptr;
    VC_CUDA(cudaDeviceSynchronize());
    int runtime_version = 0, driver_version = 0, nccl_version = 0;
    VC_CUDA(cudaRuntimeGetVersion(&runtime_version)); VC_CUDA(cudaDriverGetVersion(&driver_version));
    VC_NCCL(ncclGetVersion(&nccl_version));
    std::ostringstream report;
    report << std::setprecision(17) << "{\"schema_version\":1,\"job_id\":" << quote(o.job)
      << ",\"rank\":" << rank << ",\"device\":" << o.devices[rank]
      << ",\"status\":\"completed\",\"driver_kind\":\"native_msccl\",\"correct\":" << (all_incorrect ? "false" : "true")
      << ",\"incorrect_elements\":" << all_incorrect << ",\"gpu_released\":true,\"datatype\":" << quote(o.datatype)
      << ",\"reduction_op\":" << quote(o.reduction) << ",\"payload_bytes\":" << o.payload
      << ",\"warmup_iterations\":" << o.warmup << ",\"timed_iterations\":" << o.timed
      << ",\"timed_wall_s\":" << wall << ",\"device_name\":" << quote(properties.name)
      << ",\"compute_capability\":" << quote(std::to_string(properties.major) + "." + std::to_string(properties.minor))
      << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
      << ",\"nccl_version\":" << nccl_version << ",\"endpoints\":[";
    for (int tb = 0; !o.capture_trace && tb < blocks; tb++) {
      if (tb) report << ',';
      const auto& c = counters[size_t(tb) * 256];
      report << "{\"tb\":" << tb << ",\"step\":0,\"service_ns\":" << c.service_ns
        << ",\"dependency_wait_ns\":" << c.dependency_wait_ns << ",\"samples\":" << c.samples
        << ",\"first_start_ns\":" << c.first_start_ns << ",\"last_end_ns\":" << c.last_end_ns
        << ",\"invalid_samples\":" << c.invalid_samples << ",\"kind\":" << c.kind << ",\"channel\":" << c.channel << '}';
    }
    report << "]";
    if (o.capture_trace) report << ",\"trace_path\":" << quote(o.output + "/trace.rank-" + std::to_string(rank) + ".bin")
      << ",\"target_event_time_us\":" << target_event_us
      << ",\"reset_after_warmup\":true,\"executed_after_reset\":1,\"gpu_completed\":true,\"clock_domain\":\"globaltimer_ns\",\"clock_sync_output\":" << quote(clock_samples);
    report << "}\n";
    save(o.output + "/rank-" + std::to_string(rank) + ".json", report.str());
    std::cout << report.str();
    MPI_Finalize();
    return all_incorrect ? 6 : 0;
  } catch (const std::exception& error) {
    bool released = false;
    // Cleanup may block on failed communication. The owning rank wrapper has
    // an independent deadline and retains an uncertain release if it must kill
    // us. A successful acknowledgement requires every CUDA operation below.
    try {
      if (rank >= 0) {
        VC_CUDA(cudaDeviceSynchronize());
        if (comm && !o.capture_trace) VC_NCCL(vericclProbeAttach(comm, nullptr, 0));
        if (table) { VC_CUDA(cudaFree(table)); table = nullptr; }
        if (timing_start) { VC_CUDA(cudaEventDestroy(timing_start)); timing_start = nullptr; }
        if (timing_end) { VC_CUDA(cudaEventDestroy(timing_end)); timing_end = nullptr; }
        if (input) { VC_CUDA(cudaFree(input)); input = nullptr; }
        if (output) { VC_CUDA(cudaFree(output)); output = nullptr; }
        if (stream) { VC_CUDA(cudaStreamDestroy(stream)); stream = nullptr; }
        if (comm) { VC_NCCL(ncclCommAbort(comm)); comm = nullptr; }
        VC_CUDA(cudaDeviceSynchronize());
        released = true;
      }
    } catch (...) {}
    std::string report = "{\"job_id\":" + quote(o.job) + ",\"rank\":" + std::to_string(rank) +
      ",\"status\":\"failed\",\"correct\":false,\"gpu_released\":" + (released ? "true" : "false") +
      ",\"error\":" + quote(error.what()) + "}\n";
    std::cerr << report;
    if (rank >= 0 && !o.output.empty()) { try { save(o.output + "/rank-" + std::to_string(rank) + ".json", report); } catch (...) {} }
    // MPI_Abort itself is never used as a GPU release acknowledgement.
    if (initialized) MPI_Abort(MPI_COMM_WORLD, 4);
    return 4;
  }
}
