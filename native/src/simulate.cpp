// Native exact-mode replay of vericcl.performance.simulation.predict.
//
// Mirrors the Python reference step for step (event order, tie breaks,
// component grouping, profile selection, errors); the Python side resolves
// costs and unavailable work and passes operations sorted by identity, so an
// operation index orders exactly like its identity string.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "rational.hpp"
#include "vericcl_core.h"

namespace vericcl {
namespace {

enum Kind : int8_t { kSend = 0, kRrc = 1, kCopy = 2, kReduce = 3, kNop = 4 };
enum EventKind : int8_t { kStart = 0, kStartupComplete = 1, kComplete = 2 };
constexpr int kSharingEstimate = 4;  // source code of "initial_sharing_estimate"
constexpr int kSourceCount = 5;

struct SimulationError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Operation {
  int8_t kind = kNop;
  bool available = false;
  int64_t payload = 0;
  int32_t flow = -1;
  int32_t source = -1;
  int32_t cost_key = -1;
  std::vector<int32_t> dependencies;
  std::vector<int32_t> resources;  // sorted
  Rational not_before, alpha, beta, inverse_beta;
  bool communication() const { return kind == kSend || kind == kRrc; }
};

struct Profile {
  int32_t source = 0;
  int64_t min_bytes = 1;
  int64_t max_bytes = -1;
  bool has_membership = false;
  std::vector<int32_t> mode;       // sorted flow ids
  std::vector<int32_t> resources;  // sorted resource ids
  std::map<int32_t, Rational> rates;
  std::map<int32_t, Rational> startup;
  std::map<int32_t, std::vector<int32_t>> membership;
};

struct Input {
  std::vector<Operation> operations;
  std::vector<Rational> capacity;
  std::vector<Profile> profiles;
  std::vector<int32_t> outputs;
  int64_t cost_keys = 0;
  bool detail = true;  // record events and segments (ranking needs only T_pred)
};

struct Event {
  double time;
  int8_t kind;
  int32_t operation;
};

struct Segment {
  int32_t operation;
  double start, end, rate;
  int32_t source;
};

Input decode(const uint8_t* data, size_t size) {
  Reader reader(data, size);
  reader.expect("VCP1");
  int64_t n = reader.scalar<int64_t>();
  int64_t resources = reader.scalar<int64_t>();
  int64_t profiles = reader.scalar<int64_t>();
  int64_t outputs = reader.scalar<int64_t>();
  Input input;
  input.cost_keys = reader.scalar<int64_t>();
  input.detail = reader.scalar<int64_t>() != 0;
  auto kind = reader.array<int8_t>(n);
  auto available = reader.array<int8_t>(n);
  auto payload = reader.array<int64_t>(n);
  auto flow = reader.array<int32_t>(n);
  auto source = reader.array<int32_t>(n);
  auto cost_key = reader.array<int32_t>(n);
  auto dependency_ptr = reader.array<int64_t>(n + 1);
  auto dependency_index = reader.array<int32_t>(dependency_ptr.back());
  auto resource_ptr = reader.array<int64_t>(n + 1);
  auto resource_index = reader.array<int32_t>(resource_ptr.back());
  input.outputs = reader.array<int32_t>(outputs);
  auto profile_source = reader.array<int32_t>(profiles);
  auto profile_min = reader.array<int64_t>(profiles);
  auto profile_max = reader.array<int64_t>(profiles);
  auto profile_membership = reader.array<int8_t>(profiles);
  auto mode_ptr = reader.array<int64_t>(profiles + 1);
  auto mode_flow = reader.array<int32_t>(mode_ptr.back());
  auto profile_resource_ptr = reader.array<int64_t>(profiles + 1);
  auto profile_resource = reader.array<int32_t>(profile_resource_ptr.back());
  auto has_startup = reader.array<int8_t>(mode_ptr.back());
  auto membership_ptr = reader.array<int64_t>(mode_ptr.back() + 1);
  auto membership_index = reader.array<int32_t>(membership_ptr.back());
  int64_t text_length = reader.scalar<int64_t>();
  auto lines = reader.lines(text_length);
  if (!reader.finished()) throw std::runtime_error("trailing native buffer data");
  size_t expected = static_cast<size_t>(3 * n + resources + 2 * mode_ptr.back());
  if (lines.size() != expected) throw std::runtime_error("native rational field count differs");
  size_t line = 0;
  input.operations.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    Operation& op = input.operations[static_cast<size_t>(i)];
    op.kind = kind[i];
    op.available = available[i] != 0;
    op.payload = payload[i];
    op.flow = flow[i];
    op.source = source[i];
    op.cost_key = cost_key[i];
    op.dependencies.assign(dependency_index.begin() + dependency_ptr[i], dependency_index.begin() + dependency_ptr[i + 1]);
    op.resources.assign(resource_index.begin() + resource_ptr[i], resource_index.begin() + resource_ptr[i + 1]);
  }
  for (auto& op : input.operations) op.not_before = Rational::from_decimal(lines[line++]);
  for (auto& op : input.operations) op.alpha = Rational::from_decimal(lines[line++]);
  for (auto& op : input.operations) {
    op.beta = Rational::from_decimal(lines[line++]);
    if (op.kind != kNop && op.available) {
      if (op.beta.sign() <= 0) throw std::runtime_error("beta must be positive");
      op.inverse_beta = Rational(1) / op.beta;
    }
  }
  input.capacity.reserve(static_cast<size_t>(resources));
  for (int64_t r = 0; r < resources; ++r) input.capacity.push_back(Rational::from_decimal(lines[line++]));
  input.profiles.resize(static_cast<size_t>(profiles));
  std::vector<std::string_view> rate_lines(lines.begin() + line, lines.begin() + line + mode_ptr.back());
  line += static_cast<size_t>(mode_ptr.back());
  std::vector<std::string_view> startup_lines(lines.begin() + line, lines.begin() + line + mode_ptr.back());
  for (int64_t p = 0; p < profiles; ++p) {
    Profile& profile = input.profiles[static_cast<size_t>(p)];
    profile.source = profile_source[p];
    profile.min_bytes = profile_min[p];
    profile.max_bytes = profile_max[p];
    profile.has_membership = profile_membership[p] != 0;
    profile.resources.assign(profile_resource.begin() + profile_resource_ptr[p], profile_resource.begin() + profile_resource_ptr[p + 1]);
    for (int64_t m = mode_ptr[p]; m < mode_ptr[p + 1]; ++m) {
      int32_t flow_id = mode_flow[m];
      profile.mode.push_back(flow_id);
      profile.rates.emplace(flow_id, Rational::from_decimal(rate_lines[m]));
      if (has_startup[m]) profile.startup.emplace(flow_id, Rational::from_decimal(startup_lines[m]));
      profile.membership.emplace(flow_id, std::vector<int32_t>(membership_index.begin() + membership_ptr[m],
                                                               membership_index.begin() + membership_ptr[m + 1]));
    }
  }
  return input;
}

class Simulator {
 public:
  explicit Simulator(Input input) : in_(std::move(input)), n_(static_cast<int32_t>(in_.operations.size())) {
    degrees_.assign(n_, 0);
    successors_.assign(n_, {});
    for (int32_t i = 0; i < n_; ++i) {
      degrees_[i] = static_cast<int32_t>(op(i).dependencies.size());
      for (int32_t predecessor : op(i).dependencies) successors_[predecessor].push_back(i);
    }
    check_acyclic();
    pending_.assign(n_, false);
    starts_.assign(n_, std::nullopt);
    ends_.assign(n_, std::nullopt);
    remaining_.resize(n_);
    used_service_.assign(static_cast<size_t>(in_.cost_keys), 0);
    used_resource_.assign(in_.capacity.size(), 0);
    used_profile_.assign(in_.profiles.size(), 0);
    sources_used_.assign(kSourceCount, 0);
    for (int32_t i = 0; i < n_; ++i) {
      if (!op(i).available) continue;
      pending_[i] = true;
      ++pending_count_;
      if (degrees_[i] == 0) releases_.push({op(i).not_before, i});
    }
  }

  void run() {
    while (pending_count_ > 0 || !startup_.empty() || !active_.empty()) {
      for (auto it = startup_.begin(); it != startup_.end();) {
        if (it->second <= now_) {
          active_.insert(it->first);
          if (in_.detail) events_.push_back({now_.to_double(), kStartupComplete, it->first});
          it = startup_.erase(it);
        } else {
          ++it;
        }
      }
      std::vector<int32_t> ready;
      while (!releases_.empty() && releases_.top().time <= now_) {
        ready.push_back(releases_.top().operation);
        releases_.pop();
      }
      std::sort(ready.begin(), ready.end());
      std::map<int32_t, int32_t> launch_profile;
      {
        std::set<int32_t> present(active_.begin(), active_.end());
        for (const auto& entry : startup_) present.insert(entry.first);
        present.insert(ready.begin(), ready.end());
        std::vector<int32_t> communication;
        for (int32_t key : present)
          if (op(key).communication()) communication.push_back(key);
        for (const auto& group : components(communication)) {
          int32_t profile = profile_for(group);
          if (profile < 0) continue;
          for (int32_t key : group)
            if (in_.profiles[profile].startup.count(op(key).flow)) launch_profile[key] = profile;
        }
      }
      for (int32_t key : ready) {
        pending_[key] = false;
        --pending_count_;
        starts_[key] = now_;
        if (in_.detail) events_.push_back({now_.to_double(), kStart, key});
        if (op(key).kind == kNop) {
          complete(key, now_);
          continue;
        }
        used_service_[op(key).cost_key] = 1;
        auto found = launch_profile.find(key);
        Rational alpha;
        int32_t source;
        if (found == launch_profile.end()) {
          alpha = op(key).alpha;
          source = op(key).source;
        } else {
          const Profile& profile = in_.profiles[found->second];
          alpha = profile.startup.at(op(key).flow);
          source = profile.source;
          used_profile_[found->second] = 1;
        }
        sources_used_[source] = 1;
        remaining_[key] = Rational(op(key).payload);
        startup_[key] = now_ + alpha;
      }
      bool restart = !releases_.empty() && releases_.top().time <= now_;
      for (const auto& entry : startup_) restart = restart || entry.second <= now_;
      if (restart) continue;

      std::map<int32_t, Rational> rates;
      std::map<int32_t, int32_t> rate_sources;
      compute_rates(rates, rate_sources);
      for (const auto& entry : rate_sources) sources_used_[entry.second] = 1;
      std::map<int32_t, Rational> finish;
      std::optional<Rational> next;
      auto consider = [&next](const Rational& value) {
        if (!next || value < *next) next = value;
      };
      for (int32_t key : active_) {
        Rational time = now_ + remaining_[key] / rates.at(key);
        consider(time);
        finish.emplace(key, std::move(time));
      }
      for (const auto& entry : startup_) consider(entry.second);
      if (!releases_.empty()) consider(releases_.top().time);
      if (!next) {
        if (pending_count_ > 0) throw SimulationError("executable is stalled by unresolved work");
        break;
      }
      if (*next <= now_) throw SimulationError("model time precision cannot represent positive progress");
      Rational delta = *next - now_;
      double start_float = in_.detail ? now_.to_double() : 0.0;
      double end_float = in_.detail ? next->to_double() : 0.0;
      for (int32_t key : active_) {
        const Rational& rate = rates.at(key);
        if (in_.detail) segments_.push_back({key, start_float, end_float, rate.to_double(), rate_sources.at(key)});
        Rational left = remaining_[key] - rate * delta;
        remaining_[key] = left.sign() < 0 ? Rational(0) : std::move(left);
      }
      now_ = *next;
      std::vector<int32_t> done;
      for (int32_t key : active_)
        if (finish.at(key) == now_) done.push_back(key);
      for (int32_t key : done) {
        active_.erase(key);
        complete(key, now_);
      }
    }
  }

  void encode(Writer& out) const {
    out.scalar<int32_t>(0);
    out.text("");
    out.scalar<int64_t>(static_cast<int64_t>(events_.size()));
    std::vector<double> times;
    std::vector<int8_t> kinds;
    std::vector<int32_t> operations;
    for (const auto& event : events_) {
      times.push_back(event.time);
      kinds.push_back(event.kind);
      operations.push_back(event.operation);
    }
    out.array(times);
    out.array(kinds);
    out.array(operations);
    out.scalar<int64_t>(static_cast<int64_t>(segments_.size()));
    std::vector<int32_t> segment_op, segment_source;
    std::vector<double> segment_start, segment_end, segment_rate;
    for (const auto& segment : segments_) {
      segment_op.push_back(segment.operation);
      segment_start.push_back(segment.start);
      segment_end.push_back(segment.end);
      segment_rate.push_back(segment.rate);
      segment_source.push_back(segment.source);
    }
    out.array(segment_op);
    out.array(segment_start);
    out.array(segment_end);
    out.array(segment_rate);
    out.array(segment_source);
    out.array(used_service_);
    out.array(used_resource_);
    out.array(used_profile_);
    out.array(sources_used_);
    out.scalar<int64_t>(static_cast<int64_t>(unmeasured_.size()));
    std::vector<int64_t> pointer{0};
    std::vector<int32_t> flows;
    for (const auto& mode : unmeasured_) {
      flows.insert(flows.end(), mode.begin(), mode.end());
      pointer.push_back(static_cast<int64_t>(flows.size()));
    }
    out.array(pointer);
    out.array(flows);
    Rational t_pred(0), output_ready(0);
    for (int32_t i = 0; i < n_; ++i)
      if (ends_[i] && t_pred < *ends_[i]) t_pred = *ends_[i];
    for (int32_t key : in_.outputs)
      if (ends_[key] && output_ready < *ends_[key]) output_ready = *ends_[key];
    out.scalar<double>(t_pred.to_double());
    out.scalar<double>(output_ready.to_double());
    out.text(t_pred.to_fraction());
  }

 private:
  struct Release {
    Rational time;
    int32_t operation;
  };
  struct Later {
    bool operator()(const Release& left, const Release& right) const {
      int order = compare(left.time, right.time);
      return order != 0 ? order > 0 : left.operation > right.operation;
    }
  };

  const Operation& op(int32_t index) const { return in_.operations[static_cast<size_t>(index)]; }

  void check_acyclic() const {
    std::vector<int32_t> remaining(degrees_);
    std::vector<int32_t> queue;
    for (int32_t i = 0; i < n_; ++i)
      if (remaining[i] == 0) queue.push_back(i);
    size_t head = 0;
    while (head < queue.size()) {
      int32_t key = queue[head++];
      for (int32_t follower : successors_[key])
        if (--remaining[follower] == 0) queue.push_back(follower);
    }
    if (static_cast<int32_t>(queue.size()) != n_) throw SimulationError("executable dependency cycle");
  }

  void complete(int32_t key, const Rational& time) {
    ends_[key] = time;
    if (in_.detail) events_.push_back({time.to_double(), kComplete, key});
    for (int32_t follower : successors_[key]) {
      if (!pending_[follower]) continue;
      if (--degrees_[follower] == 0) releases_.push({op(follower).not_before, follower});
    }
  }

  // Connected components by shared resource, ordered by their first member
  // (the Python reference pops the lowest remaining identity first).
  std::vector<std::vector<int32_t>> components(const std::vector<int32_t>& keys) const {
    std::vector<int32_t> parent(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) parent[i] = static_cast<int32_t>(i);
    auto find = [&parent](int32_t x) {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    std::map<int32_t, int32_t> owner;
    for (size_t i = 0; i < keys.size(); ++i) {
      for (int32_t resource : op(keys[i]).resources) {
        auto found = owner.find(resource);
        if (found == owner.end()) {
          owner.emplace(resource, static_cast<int32_t>(i));
        } else {
          int32_t a = find(found->second), b = find(static_cast<int32_t>(i));
          if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
      }
    }
    std::map<int32_t, std::vector<int32_t>> grouped;
    for (size_t i = 0; i < keys.size(); ++i) grouped[find(static_cast<int32_t>(i))].push_back(keys[i]);
    std::vector<std::vector<int32_t>> result;
    for (auto& entry : grouped) result.push_back(std::move(entry.second));
    return result;
  }

  bool applies(const Profile& profile, const std::vector<int32_t>& group, const std::vector<int32_t>& resources) const {
    if (resources != profile.resources) return false;
    for (int32_t key : group) {
      const Operation& operation = op(key);
      if (profile.has_membership && profile.membership.at(operation.flow) != operation.resources) return false;
      if (operation.payload < profile.min_bytes) return false;
      if (profile.max_bytes >= 0 && operation.payload > profile.max_bytes) return false;
    }
    return true;
  }

  // Profiles indexed by their (sorted) mode: a lookup compares only the
  // profiles of the group's own mode instead of scanning the whole model.
  int32_t profile_for(const std::vector<int32_t>& group) {
    if (profiles_by_mode_.empty() && !in_.profiles.empty()) {
      for (size_t p = 0; p < in_.profiles.size(); ++p)
        profiles_by_mode_[in_.profiles[p].mode].push_back(static_cast<int32_t>(p));
    }
    std::vector<int32_t> mode, resources;
    for (int32_t key : group) {
      mode.push_back(op(key).flow);
      resources.insert(resources.end(), op(key).resources.begin(), op(key).resources.end());
    }
    std::sort(mode.begin(), mode.end());
    auto found = profiles_by_mode_.find(mode);
    if (found == profiles_by_mode_.end()) return -1;
    std::sort(resources.begin(), resources.end());
    resources.erase(std::unique(resources.begin(), resources.end()), resources.end());
    int32_t match = -1;
    for (int32_t p : found->second) {
      if (!applies(in_.profiles[static_cast<size_t>(p)], group, resources)) continue;
      if (match >= 0) throw SimulationError("multiple profiles apply to the same concurrency mode");
      match = p;
    }
    return match;
  }

  void compute_rates(std::map<int32_t, Rational>& rates, std::map<int32_t, int32_t>& sources) {
    std::vector<int32_t> communication;
    for (int32_t key : active_) {
      if (op(key).communication()) {
        communication.push_back(key);
      } else {
        rates.emplace(key, op(key).inverse_beta);
        sources.emplace(key, op(key).source);
      }
    }
    for (const auto& group : components(communication)) {
      std::set<int32_t> flows;
      for (int32_t key : group) flows.insert(op(key).flow);
      if (flows.size() != group.size())
        throw SimulationError("concurrent operations share a flow identity; serialize their TB work");
      int32_t profile_index = profile_for(group);
      std::set<int32_t> resources;
      for (int32_t key : group) resources.insert(op(key).resources.begin(), op(key).resources.end());
      for (int32_t resource : resources) used_resource_[resource] = 1;
      if (profile_index >= 0) {
        const Profile& profile = in_.profiles[profile_index];
        for (int32_t resource : resources) {
          Rational assigned(0);
          for (int32_t key : group)
            if (std::binary_search(op(key).resources.begin(), op(key).resources.end(), resource))
              assigned = assigned + profile.rates.at(op(key).flow);
          if (assigned > in_.capacity[resource])
            throw SimulationError("measured rate vector conflicts with resource capacity " + std::to_string(resource));
        }
        used_profile_[profile_index] = 1;
        for (int32_t key : group) {
          rates.emplace(key, profile.rates.at(op(key).flow));
          sources.emplace(key, profile.source);
        }
      } else {
        if (group.size() > 1) {
          std::vector<int32_t> mode(flows.begin(), flows.end());
          unmeasured_.insert(mode);
        }
        for (int32_t key : group) {
          const Operation& operation = op(key);
          Rational best = operation.inverse_beta;
          for (int32_t resource : operation.resources) {
            long count = 0;
            for (int32_t member : group)
              count += std::binary_search(op(member).resources.begin(), op(member).resources.end(), resource);
            Rational limit = in_.capacity[resource] / Rational(count);
            if (limit < best) best = std::move(limit);
          }
          bool shared = group.size() > 1 || best < operation.inverse_beta;
          rates.emplace(key, std::move(best));
          sources.emplace(key, shared ? kSharingEstimate : operation.source);
        }
      }
    }
  }

  Input in_;
  int32_t n_;
  std::vector<int32_t> degrees_;
  std::vector<std::vector<int32_t>> successors_;
  std::vector<bool> pending_;
  int64_t pending_count_ = 0;
  std::priority_queue<Release, std::vector<Release>, Later> releases_;
  std::map<int32_t, Rational> startup_;
  std::set<int32_t> active_;
  std::vector<Rational> remaining_;
  std::vector<std::optional<Rational>> starts_, ends_;
  Rational now_{0};
  std::vector<Event> events_;
  std::vector<Segment> segments_;
  std::vector<int8_t> used_service_, used_resource_, used_profile_, sources_used_;
  std::map<std::vector<int32_t>, std::vector<int32_t>> profiles_by_mode_;
  std::set<std::vector<int32_t>> unmeasured_;
};

}  // namespace
}  // namespace vericcl

extern "C" int vc_predict(const uint8_t* input, size_t input_size, uint8_t** output, size_t* output_size) {
  vericcl::Writer out;
  int status = 0;
  try {
    vericcl::Simulator simulator(vericcl::decode(input, input_size));
    simulator.run();
    simulator.encode(out);
  } catch (const vericcl::SimulationError& error) {
    out = vericcl::Writer();
    out.scalar<int32_t>(1);  // semantic error: same message as the Python reference
    out.text(error.what());
    status = 1;
  } catch (const std::exception& error) {
    out = vericcl::Writer();
    out.scalar<int32_t>(2);  // malformed input / internal failure
    out.text(error.what());
    status = 2;
  }
  auto& bytes = out.bytes();
  *output_size = bytes.size();
  *output = static_cast<uint8_t*>(std::malloc(bytes.size()));
  if (*output == nullptr) return 3;
  std::memcpy(*output, bytes.data(), bytes.size());
  return status;
}

extern "C" void vc_free(void* pointer) { std::free(pointer); }

extern "C" int vc_abi_version(void) { return VERICCL_CORE_ABI_VERSION; }
