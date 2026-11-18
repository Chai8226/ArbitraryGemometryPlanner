#pragma once

namespace mighty {

inline constexpr double kAdditionalYawVariationBudget = 1.0;
inline constexpr int kYawValidationIntervals = 32;

struct YawKnot {
  double yaw{0.0};
  double rate{0.0};
  double accel{0.0};
};

struct YawSample {
  double yaw{0.0};
  double rate{0.0};
  double accel{0.0};
  double jerk{0.0};
};

}  // namespace mighty
