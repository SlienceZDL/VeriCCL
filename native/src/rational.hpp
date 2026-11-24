// Exact rationals (GMP mpq_t) with the conversions the Python reference uses.
#pragma once

#include <gmp.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vericcl {

class Rational {
 public:
  Rational() { mpq_init(value_); }
  explicit Rational(long integer) {
    mpq_init(value_);
    mpq_set_si(value_, integer, 1);
  }
  Rational(const Rational& other) {
    mpq_init(value_);
    mpq_set(value_, other.value_);
  }
  Rational(Rational&& other) noexcept {
    mpq_init(value_);
    mpq_swap(value_, other.value_);
  }
  Rational& operator=(const Rational& other) {
    if (this != &other) mpq_set(value_, other.value_);
    return *this;
  }
  Rational& operator=(Rational&& other) noexcept {
    mpq_swap(value_, other.value_);
    return *this;
  }
  ~Rational() { mpq_clear(value_); }

  mpq_srcptr get() const { return value_; }
  mpq_ptr get() { return value_; }

  int sign() const { return mpq_sgn(value_); }

  // Exact value of a Python ``str(float)`` / ``str(int)`` rendering, as
  // gmpy2.mpq(str(value)) reads it: [-]digits[.digits][e[+-]digits].
  static Rational from_decimal(std::string_view text) {
    Rational result;
    if (text.empty()) throw std::invalid_argument("empty decimal");
    bool negative = false;
    size_t position = 0;
    if (text[position] == '+' || text[position] == '-') {
      negative = text[position] == '-';
      ++position;
    }
    std::string digits;
    long fraction_digits = 0;
    bool seen_point = false;
    for (; position < text.size(); ++position) {
      char character = text[position];
      if (character >= '0' && character <= '9') {
        digits.push_back(character);
        if (seen_point) ++fraction_digits;
      } else if (character == '.' && !seen_point) {
        seen_point = true;
      } else {
        break;
      }
    }
    long exponent = 0;
    if (position < text.size()) {
      if (text[position] != 'e' && text[position] != 'E') throw std::invalid_argument("bad decimal");
      ++position;
      bool exponent_negative = false;
      if (position < text.size() && (text[position] == '+' || text[position] == '-')) {
        exponent_negative = text[position] == '-';
        ++position;
      }
      if (position >= text.size()) throw std::invalid_argument("bad exponent");
      for (; position < text.size(); ++position) {
        char character = text[position];
        if (character < '0' || character > '9') throw std::invalid_argument("bad exponent");
        exponent = exponent * 10 + (character - '0');
        if (exponent > 100000) throw std::invalid_argument("exponent out of range");
      }
      if (exponent_negative) exponent = -exponent;
    }
    if (digits.empty()) throw std::invalid_argument("decimal without digits");
    mpz_t numerator, scale;
    mpz_init(numerator);
    mpz_init(scale);
    mpz_set_str(numerator, digits.c_str(), 10);
    long power = exponent - fraction_digits;
    if (power >= 0) {
      mpz_ui_pow_ui(scale, 10, static_cast<unsigned long>(power));
      mpz_mul(numerator, numerator, scale);
      mpq_set_z(result.value_, numerator);
    } else {
      mpz_ui_pow_ui(scale, 10, static_cast<unsigned long>(-power));
      mpq_set_num(result.value_, numerator);
      mpq_set_den(result.value_, scale);
      mpq_canonicalize(result.value_);
    }
    if (negative) mpq_neg(result.value_, result.value_);
    mpz_clear(numerator);
    mpz_clear(scale);
    return result;
  }

  // Correctly rounded (round-half-even) double, as CPython's int / int.
  double to_double() const {
    if (mpq_sgn(value_) == 0) return 0.0;
    mpz_t numerator, denominator, quotient, remainder, mantissa, low;
    mpz_inits(numerator, denominator, quotient, remainder, mantissa, low, nullptr);
    mpz_abs(numerator, mpq_numref(value_));
    mpz_set(denominator, mpq_denref(value_));
    long numerator_bits = static_cast<long>(mpz_sizeinbase(numerator, 2));
    long denominator_bits = static_cast<long>(mpz_sizeinbase(denominator, 2));
    long shift = 55 - (numerator_bits - denominator_bits);
    if (shift >= 0) {
      mpz_mul_2exp(numerator, numerator, static_cast<mp_bitcnt_t>(shift));
    } else {
      mpz_mul_2exp(denominator, denominator, static_cast<mp_bitcnt_t>(-shift));
    }
    mpz_tdiv_qr(quotient, remainder, numerator, denominator);
    long quotient_bits = static_cast<long>(mpz_sizeinbase(quotient, 2));
    long extra = quotient_bits - 53;  // 2 or 3 guard bits
    mpz_fdiv_q_2exp(mantissa, quotient, static_cast<mp_bitcnt_t>(extra));
    mpz_fdiv_r_2exp(low, quotient, static_cast<mp_bitcnt_t>(extra));
    unsigned long half = 1UL << (extra - 1);
    unsigned long low_value = mpz_get_ui(low);
    bool sticky = mpz_sgn(remainder) != 0;
    if (low_value > half || (low_value == half && (sticky || mpz_odd_p(mantissa)))) {
      mpz_add_ui(mantissa, mantissa, 1);
    }
    double result = std::ldexp(mpz_get_d(mantissa), static_cast<int>(extra - shift));
    if (mpq_sgn(value_) < 0) result = -result;
    mpz_clears(numerator, denominator, quotient, remainder, mantissa, low, nullptr);
    return result;
  }

  std::string to_fraction() const {
    char* numerator = mpz_get_str(nullptr, 10, mpq_numref(value_));
    char* denominator = mpz_get_str(nullptr, 10, mpq_denref(value_));
    std::string text = std::string(numerator) + "/" + denominator;
    void (*release)(void*, size_t);
    mp_get_memory_functions(nullptr, nullptr, &release);
    release(numerator, std::char_traits<char>::length(numerator) + 1);
    release(denominator, std::char_traits<char>::length(denominator) + 1);
    return text;
  }

 private:
  mpq_t value_;
};

inline Rational operator+(const Rational& left, const Rational& right) {
  Rational result;
  mpq_add(result.get(), left.get(), right.get());
  return result;
}
inline Rational operator-(const Rational& left, const Rational& right) {
  Rational result;
  mpq_sub(result.get(), left.get(), right.get());
  return result;
}
inline Rational operator*(const Rational& left, const Rational& right) {
  Rational result;
  mpq_mul(result.get(), left.get(), right.get());
  return result;
}
inline Rational operator/(const Rational& left, const Rational& right) {
  Rational result;
  mpq_div(result.get(), left.get(), right.get());
  return result;
}
inline int compare(const Rational& left, const Rational& right) { return mpq_cmp(left.get(), right.get()); }
inline bool operator<(const Rational& left, const Rational& right) { return compare(left, right) < 0; }
inline bool operator<=(const Rational& left, const Rational& right) { return compare(left, right) <= 0; }
inline bool operator>(const Rational& left, const Rational& right) { return compare(left, right) > 0; }
inline bool operator==(const Rational& left, const Rational& right) { return mpq_equal(left.get(), right.get()) != 0; }

}  // namespace vericcl
