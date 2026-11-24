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

// ---------------------------------------------------------------------------
// Batch session: one MPI launch runs many probe modes. Each mode gets its own
// NCCL communicator over its member ranks (MPI_Comm_split of the session) and
// its MSCCL XML; each of its launches is an independent barrier-separated
// episode with its own warmup and timed iterations, reported exactly like a
// single-launch probe. Results are JSON lines per (mode, session rank),
// published by rename without a per-file fsync.
struct BatchJob {
  std::string key, xml, datatype = "float32", reduction = "sum";
  size_t payload = 0;
  int warmup = 5, timed = 20, launches = 1;
  double atol = 1e-5, rtol = 1e-5;
  std::vector<int> members;
  std::vector<Flow> flows;
};

std::map<std::string, std::string> plan_fields(const std::string& line) {
  std::map<std::string, std::string> result;
  std::stringstream stream(line);
  std::string token;
  stream >> token;  // record kind
  while (stream >> token) {
    auto equal = token.find('=');
    if (equal == std::string::npos) throw std::runtime_error("invalid batch plan token: " + token);
    result[token.substr(0, equal)] = token.substr(equal + 1);
  }
  return result;
}

void publish_plain(const std::string& path, const std::string& text) {
  std::string temporary = path + ".partial";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out << text;
  out.close();
  if (!out) throw std::runtime_error("cannot write batch evidence");
  if (rename(temporary.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot publish batch evidence");
}

// Runs every launch of one job on an already initialized communicator; the
// stream, buffers and counter table are per job (payload sizes differ).
std::string run_batch_job(const BatchJob& job, MPI_Comm comm, ncclComm_t nccl, int rank, int device,
                          const cudaDeviceProp& properties, int runtime_version, int driver_version) {
  int ranks = int(job.members.size());
  auto type = datatype(job.datatype);
  auto op = reduction(job.reduction);
  if (job.payload % datatype_bytes(type)) throw std::runtime_error("payload is not a whole datatype element");
  int blocks = 0;
  for (const auto& f : job.flows) {
    if (f.src >= ranks || f.dst >= ranks || f.channel >= 16 ||
        (f.kind != "send" && f.kind != "rrc" && f.kind != "copy" && f.kind != "reduce") ||
        ((f.src == f.dst) != (f.kind == "copy" || f.kind == "reduce")))
      throw std::runtime_error("invalid flow endpoints or primitive");
    if (f.src == rank || f.dst == rank) blocks++;
  }
  if (!blocks || blocks > 216) throw std::runtime_error("invalid active-rank TB count");
  cudaStream_t stream = nullptr;
  void *input = nullptr, *output = nullptr;
  int* alignment = nullptr;
  VericclProbeCounter* table = nullptr;
  bool attached = false;
  std::string lines;
  try {
    VC_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t elements = job.payload / datatype_bytes(type);
    size_t bytes = job.payload * job.flows.size();
    VC_CUDA(cudaMalloc(&input, bytes)); VC_CUDA(cudaMalloc(&output, bytes));
    size_t entries = size_t(blocks) * 256;
    VC_CUDA(cudaMalloc(&table, entries * sizeof(*table)));
    VC_CUDA(cudaMemsetAsync(table, 0, entries * sizeof(*table), stream));
    for (size_t slot = 0; slot < job.flows.size(); slot++)
      fill_values((char*)input + slot * job.payload, elements, type, value(rank, slot), stream);
    VC_CUDA(cudaStreamSynchronize(stream));
    VC_NCCL(vericclProbeAttach(nccl, table, entries));
    attached = true;
    int nccl_version = 0;
    VC_NCCL(ncclGetVersion(&nccl_version));
    // Iterations are aligned on the GPU, not the host: a one-element
    // allreduce on the probe communicator precedes each probe on the same
    // stream, so every GPU starts the probe right after all peers arrive.
    // A host MPI barrier per iteration costs 12-47 ms once the session spans
    // two hosts (TCP progress), which hides sub-millisecond probes.
    VC_CUDA(cudaMalloc(&alignment, sizeof(int)));
    VC_CUDA(cudaMemsetAsync(alignment, 0, sizeof(int), stream));
    // Default: ranks align once per launch (barrier above); iterations then
    // run back to back, and the flows of one probe synchronize through their
    // own connections. Per-iteration alignment ("allreduce" on the GPU or
    // "mpi" on the host) left the measured GPU service unchanged on 2x8 V100
    // but cost 6x more wall time for multi-flow inter-host modes.
    const char* align_env = getenv("VERICCL_BATCH_ALIGN");
    const std::string align = align_env ? align_env : "none";
    auto run = [&]() {
      for (size_t slot = 0; slot < job.flows.size(); slot++)
        fill_values((char*)output + slot * job.payload, elements, type, accumulator(rank, slot), stream);
      if (align == "allreduce") {
        VC_NCCL(ncclAllReduce(alignment, alignment, 1, ncclInt32, ncclSum, nccl, stream));
      } else if (align == "mpi") {
        VC_CUDA(cudaStreamSynchronize(stream));
        if (MPI_Barrier(comm) != MPI_SUCCESS) throw std::runtime_error("MPI probe barrier failed");
      }
      VC_NCCL(ncclCustomCollective(input, output, elements * job.flows.size(), type, op, 0, nccl, stream));
    };
    std::vector<VericclProbeCounter> counters(entries);
    for (int launch = 1; launch <= job.launches; launch++) {
      if (MPI_Barrier(comm) != MPI_SUCCESS) throw std::runtime_error("MPI launch barrier failed");
      for (int i = 0; i < job.warmup; i++) run();
      VC_CUDA(cudaStreamSynchronize(stream));
      VC_CUDA(cudaMemsetAsync(table, 0, entries * sizeof(*table), stream));
      VC_CUDA(cudaStreamSynchronize(stream));
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < job.timed; i++) run();
      VC_CUDA(cudaStreamSynchronize(stream));
      double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      VC_CUDA(cudaMemcpy(counters.data(), table, entries * sizeof(*table), cudaMemcpyDeviceToHost));
      unsigned long long incorrect = 0;
      for (size_t slot = 0; slot < job.flows.size(); slot++) {
        const auto& f = job.flows[slot];
        if (f.dst != rank) continue;
        double expected = value(f.src, slot);
        if (f.kind == "rrc") expected = combine(expected, value(rank, slot), op);
        if (f.kind == "reduce") expected = combine(expected, accumulator(rank, slot), op);
        incorrect += check_values((char*)output + slot * job.payload, elements, type, expected, job.atol, job.rtol, stream);
      }
      for (size_t slot = 0; slot < job.flows.size(); slot++)
        incorrect += check_values((char*)input + slot * job.payload, elements, type, value(rank, slot), job.atol, job.rtol, stream);
      unsigned long long all_incorrect = 0;
      if (MPI_Allreduce(&incorrect, &all_incorrect, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm) != MPI_SUCCESS)
        throw std::runtime_error("MPI correctness agreement failed");
      std::ostringstream report;
      report << std::setprecision(17) << "{\"schema_version\":1,\"job_id\":" << quote(job.key)
        << ",\"launch\":" << launch << ",\"rank\":" << rank << ",\"device\":" << device
        << ",\"status\":\"completed\",\"driver_kind\":\"native_msccl\",\"correct\":" << (all_incorrect ? "false" : "true")
        << ",\"incorrect_elements\":" << all_incorrect << ",\"gpu_released\":true,\"datatype\":" << quote(job.datatype)
        << ",\"reduction_op\":" << quote(job.reduction) << ",\"payload_bytes\":" << job.payload
        << ",\"warmup_iterations\":" << job.warmup << ",\"timed_iterations\":" << job.timed
        << ",\"timed_wall_s\":" << wall << ",\"device_name\":" << quote(properties.name)
        << ",\"compute_capability\":" << quote(std::to_string(properties.major) + "." + std::to_string(properties.minor))
        << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
        << ",\"nccl_version\":" << nccl_version << ",\"endpoints\":[";
      // Every executed step of every TB (256 step slots per TB): a chained
      // probe flow runs several steps on one TB.
      bool first_endpoint = true;
      for (int tb = 0; tb < blocks; tb++) for (int step = 0; step < 256; step++) {
        const auto& c = counters[size_t(tb) * 256 + step];
        if (step && !c.samples && !c.invalid_samples) continue;
        if (!first_endpoint) report << ',';
        first_endpoint = false;
        report << "{\"tb\":" << tb << ",\"step\":" << step << ",\"service_ns\":" << c.service_ns
          << ",\"dependency_wait_ns\":" << c.dependency_wait_ns << ",\"samples\":" << c.samples
          << ",\"first_start_ns\":" << c.first_start_ns << ",\"last_end_ns\":" << c.last_end_ns
          << ",\"invalid_samples\":" << c.invalid_samples << ",\"kind\":" << c.kind << ",\"channel\":" << c.channel << '}';
      }
      report << "]}\n";
      lines += report.str();
    }
    VC_NCCL(vericclProbeAttach(nccl, nullptr, 0)); attached = false;
    VC_CUDA(cudaFree(alignment)); alignment = nullptr;
    VC_CUDA(cudaFree(table)); table = nullptr;
    VC_CUDA(cudaFree(input)); input = nullptr;
    VC_CUDA(cudaFree(output)); output = nullptr;
    VC_CUDA(cudaStreamDestroy(stream)); stream = nullptr;
    VC_CUDA(cudaDeviceSynchronize());
    return lines;
  } catch (...) {
    // The caller aborts the communicator; release this job's resources.
    try {
      cudaDeviceSynchronize();
      if (attached) vericclProbeAttach(nccl, nullptr, 0);
      if (alignment) cudaFree(alignment);
      if (table) cudaFree(table);
      if (input) cudaFree(input);
      if (output) cudaFree(output);
      if (stream) cudaStreamDestroy(stream);
      cudaDeviceSynchronize();
    } catch (...) {}
    throw;
  }
}

// Session-wide waits must not busy-poll: ranks idle in a blocking
// MPI_Barrier/Allreduce (TCP progress) slowed the active ranks' NCCL network
// proxies from 1-12 ms to 45-92 ms per inter-host iteration on 2x8 V100.
void polite_wait(MPI_Request& request, const char* what) {
  int done = 0;
  while (!done) {
    if (MPI_Test(&request, &done, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error(what);
    if (!done) usleep(500);
  }
}

void polite_barrier(MPI_Comm comm) {
  MPI_Request request;
  if (MPI_Ibarrier(comm, &request) != MPI_SUCCESS) throw std::runtime_error("MPI session barrier failed");
  polite_wait(request, "MPI session barrier failed");
}

int polite_max(int value, MPI_Comm comm) {
  int result = 0;
  MPI_Request request;
  if (MPI_Iallreduce(&value, &result, 1, MPI_INT, MPI_MAX, comm, &request) != MPI_SUCCESS)
    throw std::runtime_error("MPI session agreement failed");
  polite_wait(request, "MPI session agreement failed");
  return result;
}

// One communicator per distinct (XML, members, flows); consecutive jobs that
// differ only in payload size reuse it. Every rank reads the same plan, so
// all ranks agree on when the communicator is rebuilt.
struct SessionComm {
  std::string signature;
  MPI_Comm sub = MPI_COMM_NULL;
  ncclComm_t nccl = nullptr;
  int position = -1;
  bool broken = false;
};

std::string job_signature(const BatchJob& job) {
  std::ostringstream text;
  text << job.xml << '|';
  for (int m : job.members) text << m << ',';
  text << '|';
  for (const auto& f : job.flows) text << f.kind << ':' << f.src << ':' << f.dst << ':' << f.channel << ',';
  return text.str();
}

void release_comm(SessionComm& state) {
  if (state.nccl) {
    if (state.broken) ncclCommAbort(state.nccl);
    else ncclCommDestroy(state.nccl);
    state.nccl = nullptr;
  }
  if (state.sub != MPI_COMM_NULL) MPI_Comm_free(&state.sub);
  state.position = -1;
  state.broken = false;
}

int run_batch(int argc, char** argv) {
  std::string plan_path, output;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string key(argv[i]);
    if (key == "--plan") plan_path = argv[i + 1];
    else if (key == "--output-dir") output = argv[i + 1];
    else throw std::runtime_error("unknown batch option: " + key);
  }
  if (plan_path.empty() || output.empty()) throw std::runtime_error("batch requires --plan and --output-dir");
  std::ifstream plan(plan_path);
  if (!plan) throw std::runtime_error("cannot read batch plan");
  std::vector<int> devices;
  std::vector<BatchJob> jobs;
  std::string line;
  while (std::getline(plan, line)) {
    if (line.empty()) continue;
    auto fields = plan_fields(line);
    if (line.rfind("world ", 0) == 0) {
      for (auto& v : split(fields.at("devices"), ',')) devices.push_back(int(integer(v)));
      continue;
    }
    if (line.rfind("job ", 0) != 0) throw std::runtime_error("unknown batch plan record");
    BatchJob job;
    job.key = fields.at("key"); job.xml = fields.at("xml");
    job.payload = integer(fields.at("payload"));
    job.datatype = fields.at("datatype"); job.reduction = fields.at("reduction");
    job.warmup = int(integer(fields.at("warmup"))); job.timed = int(integer(fields.at("timed")));
    job.launches = int(integer(fields.at("launches")));
    job.atol = std::stod(fields.at("atol")); job.rtol = std::stod(fields.at("rtol"));
    for (auto& v : split(fields.at("members"), ',')) job.members.push_back(int(integer(v)));
    for (auto& v : split(fields.at("flows"), ',')) {
      auto parts = split(v, ':');
      if (parts.size() != 4) throw std::runtime_error("invalid batch flow");
      job.flows.push_back({parts[0], int(integer(parts[1])), int(integer(parts[2])), int(integer(parts[3]))});
    }
    if (!job.payload || job.timed < 1 || job.launches < 1 || job.members.empty() || job.flows.empty())
      throw std::runtime_error("incomplete batch job: " + job.key);
    jobs.push_back(std::move(job));
  }
  int rank = -1, ranks = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int completed = 0, failed = 0;
  bool released = true;
  try {
    if (ranks != int(devices.size())) throw std::runtime_error("MPI session size differs from the batch plan");
    VC_CUDA(cudaSetDevice(devices[rank]));
    cudaDeviceProp properties;
    VC_CUDA(cudaGetDeviceProperties(&properties, devices[rank]));
    int runtime_version = 0, driver_version = 0;
    VC_CUDA(cudaRuntimeGetVersion(&runtime_version)); VC_CUDA(cudaDriverGetVersion(&driver_version));
    SessionComm state;
    std::string init_error;
    auto failure_line = [](const BatchJob& job, int position, const std::string& message) {
      return "{\"schema_version\":1,\"job_id\":" + quote(job.key) + ",\"rank\":" + std::to_string(position) +
             ",\"status\":\"failed\",\"correct\":false,\"gpu_released\":true,\"error\":" + quote(message) + "}\n";
    };
    for (const auto& job : jobs) {
      std::string signature = job_signature(job);
      int broken_any = polite_max(state.broken ? 1 : 0, MPI_COMM_WORLD);
      if (signature != state.signature || broken_any) {
        state.broken = state.broken || broken_any;
        release_comm(state);
        polite_barrier(MPI_COMM_WORLD);
        int position = -1;
        for (size_t i = 0; i < job.members.size(); i++)
          if (job.members[i] == rank) position = int(i);
        if (MPI_Comm_split(MPI_COMM_WORLD, position >= 0 ? 0 : MPI_UNDEFINED, position, &state.sub) != MPI_SUCCESS)
          throw std::runtime_error("MPI session split failed");
        state.signature = signature;
        state.position = position;
        init_error.clear();
        if (position >= 0) {
          try {
            if (setenv("MSCCL_XML_FILES", job.xml.c_str(), 1) != 0 || setenv("VERICCL_TRACE_ENABLE", "0", 1) != 0)
              throw std::runtime_error("cannot set probe environment");
            ncclUniqueId id;
            if (position == 0) VC_NCCL(ncclGetUniqueId(&id));
            if (MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, state.sub) != MPI_SUCCESS)
              throw std::runtime_error("MPI unique-id broadcast failed");
            VC_NCCL(ncclCommInitRank(&state.nccl, int(job.members.size()), id, position));
          } catch (const std::exception& error) {
            init_error = error.what();
            state.broken = true;
          }
        }
      }
      if (state.position >= 0) {
        std::string lines;
        if (!state.nccl) {
          failed++;
          lines = failure_line(job, state.position, init_error.empty() ? "communicator unavailable" : init_error);
        } else {
          try {
            lines = run_batch_job(job, state.sub, state.nccl, state.position, devices[rank], properties,
                                  runtime_version, driver_version);
            completed++;
          } catch (const std::exception& error) {
            failed++;
            state.broken = true;
            lines = failure_line(job, state.position, error.what());
          }
        }
        publish_plain(output + "/" + job.key + ".w" + std::to_string(rank) + ".jsonl", lines);
      }
      polite_barrier(MPI_COMM_WORLD);
    }
    release_comm(state);
    VC_CUDA(cudaDeviceSynchronize());
  } catch (const std::exception& error) {
    released = false;
    try { cudaDeviceSynchronize(); released = true; } catch (...) {}
    std::string report = "{\"schema_version\":1,\"rank\":" + std::to_string(rank) + ",\"status\":\"failed\",\"gpu_released\":" +
      (released ? "true" : "false") + ",\"error\":" + quote(error.what()) + "}\n";
    std::cerr << report;
    try { save(output + "/rank-" + std::to_string(rank) + ".json", report); } catch (...) {}
    MPI_Abort(MPI_COMM_WORLD, 4);
    return 4;
  }
  std::string receipt = "{\"schema_version\":1,\"rank\":" + std::to_string(rank) +
    ",\"status\":\"completed\",\"driver_kind\":\"native_msccl\",\"gpu_released\":true,\"jobs_completed\":" +
    std::to_string(completed) + ",\"jobs_failed\":" + std::to_string(failed) + "}\n";
  save(output + "/rank-" + std::to_string(rank) + ".json", receipt);
  std::cout << receipt;
  MPI_Finalize();
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "batch") {
    try {
      return run_batch(argc, argv);
    } catch (const std::exception& error) {
      std::cerr << "{\"status\":\"failed\",\"error\":" << quote(error.what()) << "}\n";
      return 4;
    }
  }
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
    bool first_endpoint = true;
    for (int tb = 0; !o.capture_trace && tb < blocks; tb++) for (int step = 0; step < 256; step++) {
      const auto& c = counters[size_t(tb) * 256 + step];
      if (step && !c.samples && !c.invalid_samples) continue;
      if (!first_endpoint) report << ',';
      first_endpoint = false;
      report << "{\"tb\":" << tb << ",\"step\":" << step << ",\"service_ns\":" << c.service_ns
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
