// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace mighty::polymap
{

inline std::array<float, 3> distinctVisualizationColor(size_t polyhedron_id)
{
  constexpr double golden_ratio_conjugate = 0.6180339887498949;
  const double hue =
      std::fmod(0.08 + static_cast<double>(polyhedron_id) *
                           golden_ratio_conjugate,
                1.0);
  constexpr double saturation = 0.72;
  constexpr double value = 0.95;
  const double scaled_hue = hue * 6.0;
  const int sector = static_cast<int>(std::floor(scaled_hue));
  const double fraction = scaled_hue - static_cast<double>(sector);
  const double p = value * (1.0 - saturation);
  const double q = value * (1.0 - saturation * fraction);
  const double t = value * (1.0 - saturation * (1.0 - fraction));

  switch (sector % 6)
  {
    case 0:
      return {static_cast<float>(value), static_cast<float>(t),
              static_cast<float>(p)};
    case 1:
      return {static_cast<float>(q), static_cast<float>(value),
              static_cast<float>(p)};
    case 2:
      return {static_cast<float>(p), static_cast<float>(value),
              static_cast<float>(t)};
    case 3:
      return {static_cast<float>(p), static_cast<float>(q),
              static_cast<float>(value)};
    case 4:
      return {static_cast<float>(t), static_cast<float>(p),
              static_cast<float>(value)};
    default:
      return {static_cast<float>(value), static_cast<float>(p),
              static_cast<float>(q)};
  }
}

}  // namespace mighty::polymap
