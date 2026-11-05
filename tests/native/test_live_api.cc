#include "vericcl.h"
#include "contract.h"
#include <cassert>
#include <iostream>

using namespace vericcl;
template<class F> void rejects(F fn) {
  bool failed = false;
  try { fn(); } catch (const std::exception&) { failed = true; }
  assert(failed);
}
int main() {
  assert(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  rejects([] { strict_json("{\"a\":1,\"a\":1}"); });
  rejects([] { strict_json("{\"a\":{\"b\":1,\"b\":2}}"); });
  auto a = strict_json("{\"a\":{\"b\":1},\"b\":{\"b\":2}}");
  assert(a["a"]["b"] == 1);
  rejects([] { strict_json("{\"a\":NaN}"); });
  rejects([] { exact_keys(strict_json("{\"a\":1,\"typo\":2}"), {"a"}); });
  BufferGeometry ag{64, 128, 0, 64, true};
  auto base = reinterpret_cast<void*>(uintptr_t(4096));
  auto part = reinterpret_cast<void*>(uintptr_t(4160));
  validate_buffers(ag, part, 64, base, 128);
  rejects([&] { validate_buffers(ag, base, 64, base, 128); });
  rejects([&] { validate_buffers(ag, part, 63, base, 128); });
  BufferGeometry rs{128, 64, 64, 0, true};
  validate_buffers(rs, base, 128, part, 64);
  BufferGeometry out{64, 64, 0, 0, false};
  rejects([&] { validate_buffers(out, base, 64, base, 64); });
  validate_buffers(out, base, 64, reinterpret_cast<void*>(uintptr_t(8192)), 64);
  auto until = Clock::now() + std::chrono::milliseconds(10);
  assert(child_deadline(until, 1) == until);
  rejects([] { finite_positive(0, "timeout"); });
  std::cout << "native CPU contracts passed; no GPU execution\n";
}
