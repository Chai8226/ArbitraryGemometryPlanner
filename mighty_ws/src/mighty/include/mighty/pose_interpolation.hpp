#pragma once

#include <algorithm>

namespace mighty {

inline double quinticPosePhase(double phase) {
  const double t = std::clamp(phase, 0.0, 1.0);
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}

}  // namespace mighty
