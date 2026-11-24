// Little-endian flat buffers shared with vericcl/native/*.py (numpy tobytes).
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vericcl {

class Reader {
 public:
  Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  template <typename T>
  T scalar() {
    T value;
    take(&value, sizeof(T));
    return value;
  }

  template <typename T>
  std::vector<T> array(int64_t count) {
    if (count < 0) throw std::runtime_error("negative array length");
    std::vector<T> values(static_cast<size_t>(count));
    if (count) take(values.data(), sizeof(T) * static_cast<size_t>(count));
    return values;
  }

  void expect(std::string_view magic) {
    if (size_ - offset_ < magic.size() || std::memcmp(data_ + offset_, magic.data(), magic.size()) != 0)
      throw std::runtime_error("unexpected native buffer format");
    offset_ += magic.size();
  }

  // Text fields, each terminated by '\n' (so empty fields are unambiguous).
  std::vector<std::string_view> lines(int64_t length) {
    if (length < 0 || static_cast<size_t>(length) > size_ - offset_) throw std::runtime_error("truncated text");
    std::string_view text(reinterpret_cast<const char*>(data_ + offset_), static_cast<size_t>(length));
    offset_ += static_cast<size_t>(length);
    std::vector<std::string_view> result;
    size_t start = 0;
    while (start < text.size()) {
      size_t end = text.find('\n', start);
      if (end == std::string_view::npos) throw std::runtime_error("unterminated text field");
      result.push_back(text.substr(start, end - start));
      start = end + 1;
    }
    return result;
  }

  bool finished() const { return offset_ == size_; }

 private:
  void take(void* target, size_t bytes) {
    if (bytes > size_ - offset_) throw std::runtime_error("truncated native buffer");
    std::memcpy(target, data_ + offset_, bytes);
    offset_ += bytes;
  }

  const uint8_t* data_;
  size_t size_;
  size_t offset_ = 0;
};

class Writer {
 public:
  template <typename T>
  void scalar(T value) {
    append(&value, sizeof(T));
  }
  template <typename T>
  void array(const std::vector<T>& values) {
    if (!values.empty()) append(values.data(), sizeof(T) * values.size());
  }
  void text(const std::string& value) {
    scalar<int64_t>(static_cast<int64_t>(value.size()));
    append(value.data(), value.size());
  }
  std::vector<uint8_t>& bytes() { return bytes_; }

 private:
  void append(const void* source, size_t size) {
    const uint8_t* begin = static_cast<const uint8_t*>(source);
    bytes_.insert(bytes_.end(), begin, begin + size);
  }
  std::vector<uint8_t> bytes_;
};

}  // namespace vericcl
