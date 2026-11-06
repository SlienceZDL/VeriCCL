#ifndef VERICCL_CONTROL_TRANSPORT_H
#define VERICCL_CONTROL_TRANSPORT_H
#include "contract.h"
#include <memory>

namespace vericcl {
struct Socket {
  int fd = -1;
  Socket() = default;
  explicit Socket(int value): fd(value) {}
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  Socket(Socket&& other) noexcept: fd(other.fd) { other.fd = -1; }
  Socket& operator=(Socket&& other) noexcept;
  ~Socket();
};
void send_json(int fd, const Json&, Deadline);
Json receive_json(int fd, Deadline);
class ControlTransport {
  int rank_, ranks_;
  std::string group_, session_;
  uint64_t sequence_ = 0;
  bool broken_ = false;
  std::string address_;
  uint16_t port_;
  struct Coordinator;
  std::unique_ptr<Coordinator> coordinator_;
 public:
  ControlTransport(int rank, int ranks, const std::string& group, const std::string& session,
      const std::string& address, const std::string& bind, uint16_t port, Deadline);
  ~ControlTransport();
  Json gather(const std::string& phase, const Json& payload, Deadline);
  /* CPU test injection only. Normal construction never drops replies. */
  void inject_lost_replies_for_test(const std::string& phase, unsigned count);
  bool broken() const { return broken_; }
  uint64_t sequence() const { return sequence_; }
};
}
#endif
