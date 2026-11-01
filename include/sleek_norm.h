#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// Remove the optional flag so positional arguments keep their existing indices.
static inline const char* sleek_parse_norm_flag(int& argc, char* argv[])
{
  const char* norm = "linf";
  int positional = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-L2") == 0) {
      if (std::strcmp(norm, "-L2") == 0) {
        std::fprintf(stderr, "ERROR: -L2 may only be specified once\n");
        std::exit(EXIT_FAILURE);
      }
      norm = "-L2";
    } else {
      argv[positional++] = argv[i];
    }
  }
  argc = positional;
  argv[argc] = nullptr;
  return norm;
}

// Convert a global absolute L2 budget to a conservative point-wise budget.
// Both compression and decompression must use the same norm and element count.
template <typename F>
static inline F sleek_norm_error_bound(const F bound, const long long count,
                                      const char* norm)
{
  if (std::strcmp(norm, "linf") == 0) return bound;
  if (std::strcmp(norm, "-L2") != 0) {
    std::fprintf(stderr, "ERROR: optional flag must be -L2\n");
    std::exit(EXIT_FAILURE);
  }
  if (count <= 0 || !std::isfinite(bound) || bound <= 0) {
    std::fprintf(stderr, "ERROR: -L2 requires a positive finite error bound and element count\n");
    std::exit(EXIT_FAILURE);
  }
  const long double divisor = std::sqrt(static_cast<long double>(count));
  // Round the divisor up and the result down to avoid exceeding the L2 budget.
  const long double limit = static_cast<long double>(bound) /
      std::nextafter(divisor, std::numeric_limits<long double>::infinity());
  F result = static_cast<F>(limit);
  if (static_cast<long double>(result) > limit) result = std::nextafter(result, F(0));
  // The codec derives its quantization step from the exponent. Subnormal
  // bounds cannot express a smaller step, so reject rather than overpromise.
  if (result < std::numeric_limits<F>::min()) {
    std::fprintf(stderr, "ERROR: -L2 per-value error bound is too small for this precision\n");
    std::exit(EXIT_FAILURE);
  }
  return result;
}
