#ifndef VERICCL_EXECUTION_CONTRACT_H
#define VERICCL_EXECUTION_CONTRACT_H
#include "vericcl.h"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vericcl {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
inline void finite_positive(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(std::string(name) + " must be finite and positive");
}
inline void validate_execution_config(const VericclConfig& config) {
  if(config.reserved || config.background_tuning_task_timeout_s!=0 || config.recovery_timeout_s!=0 ||
      config.max_retries || config.fault_inject_attempts || config.fault_inject_request)
    throw std::invalid_argument("removed runtime-update and recovery fields are reserved and must be zero");
  for(auto seconds:{config.request_prepare_timeout_s,config.execution_attempt_timeout_s,
      config.version_prepare_timeout_s,config.control_phase_timeout_s,config.close_cleanup_timeout_s})
    finite_positive(seconds,"timeout");
}
inline Deadline child_deadline(Deadline parent, double seconds) {
  finite_positive(seconds, "deadline");
  return std::min(parent, Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds)));
}
inline Deadline after(double seconds) { return child_deadline(Deadline::max(), seconds); }
inline double elapsed_us(Deadline start) { return std::chrono::duration<double, std::micro>(Clock::now()-start).count(); }
inline Json strict_json(const std::string& source, size_t limit = 4 * 1024 * 1024) {
  if (source.size() > limit) throw std::invalid_argument("JSON exceeds configured byte limit");
  std::vector<std::set<std::string>> keys;
  return Json::parse(source, [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth > 64) throw std::invalid_argument("JSON nesting exceeds 64");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key && (keys.empty() || !keys.back().insert(value.get<std::string>()).second))
      throw std::invalid_argument("duplicate JSON key");
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  });
}
inline void exact_keys(const Json& value, const std::set<std::string>& allowed, const std::set<std::string>& required = {}) {
  if (!value.is_object()) throw std::invalid_argument("expected JSON object");
  for (auto it = value.begin(); it != value.end(); ++it)
    if (!allowed.count(it.key())) throw std::invalid_argument("unknown field: " + it.key());
  for (const auto& key : required) if (!value.contains(key)) throw std::invalid_argument("missing field: " + key);
}
inline uint64_t integer(const Json& value, const char* name, uint64_t maximum = INT64_MAX) {
  if ((!value.is_number_unsigned() && !value.is_number_integer()) ||
      (value.is_number_integer() && !value.is_number_unsigned() && value.get<int64_t>() < 0))
    throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
  auto result = value.get<uint64_t>();
  if (result > maximum) throw std::invalid_argument(std::string(name) + " exceeds limit");
  return result;
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > uint64_t(INT64_MAX) / b) throw std::invalid_argument("byte size overflow");
  return a*b;
}
inline std::string read_file(const std::string& path, size_t limit = 16*1024*1024) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::invalid_argument("cannot read file: " + path);
  in.seekg(0, std::ios::end); auto size = in.tellg();
  if (size < 0 || uint64_t(size) > limit) throw std::invalid_argument("file exceeds size limit");
  in.seekg(0); std::string result(size_t(size), '\0');
  if (size && !in.read(result.data(), size)) throw std::invalid_argument("incomplete file read");
  return result;
}
/* SHA-256, FIPS 180-4 compression function. Fixed test vectors are exercised
 * by the CPU contract test and manifest interoperability tests. */
inline std::string sha256(const std::string& input) {
  static const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::vector<uint8_t> data(input.begin(), input.end()); uint64_t bits = uint64_t(data.size())*8;
  data.push_back(0x80); while (data.size()%64 != 56) data.push_back(0);
  for (int i=7;i>=0;--i) data.push_back(uint8_t(bits>>(i*8)));
  auto rr=[](uint32_t v,int n){return (v>>n)|(v<<(32-n));};
  for (size_t off=0;off<data.size();off+=64) {
    uint32_t w[64];
    for(int i=0;i<16;i++) w[i]=(uint32_t(data[off+4*i])<<24)|(uint32_t(data[off+4*i+1])<<16)|(uint32_t(data[off+4*i+2])<<8)|data[off+4*i+3];
    for(int i=16;i<64;i++) w[i]=w[i-16]+(rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>10));
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for(int i=0;i<64;i++) {uint32_t t=z+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+k[i]+w[i];uint32_t u=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c));z=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
  std::ostringstream out; out<<std::hex<<std::setfill('0');for(auto v:h)out<<std::setw(8)<<v;return out.str();
}
struct BufferGeometry {
  uint64_t input_bytes, output_bytes, output_offset, input_offset;
  bool inplace;
};
inline void validate_buffers(const BufferGeometry& g, const void* input, uint64_t input_size, const void* output, uint64_t output_size) {
  if ((g.input_bytes && !input) || (g.output_bytes && !output) || input_size<g.input_bytes || output_size<g.output_bytes)
    throw std::invalid_argument("buffer is null or smaller than checked schedule geometry");
  auto i = reinterpret_cast<uintptr_t>(input), o = reinterpret_cast<uintptr_t>(output);
  if (g.input_bytes>UINTPTR_MAX-i || g.output_bytes>UINTPTR_MAX-o) throw std::invalid_argument("buffer address overflow");
  if (g.inplace) {
    if (i < g.input_offset || o < g.output_offset || i-g.input_offset != o-g.output_offset)
      throw std::invalid_argument("in-place alias does not match collective rank partition");
  } else if (g.input_bytes && g.output_bytes && i<o+g.output_bytes && o<i+g.input_bytes)
    throw std::invalid_argument("out-of-place input/output overlap");
}
} // namespace vericcl
#endif
