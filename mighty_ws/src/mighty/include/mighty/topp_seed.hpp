#pragma once

#include <vector>

#include <Eigen/Core>

namespace mighty {

struct ToppLimits {
  double v_max{2.0};
  double a_max{5.0};
  double j_max{10.0};
  double w_max{1.0};
  double yaw_acc_max{2.0};
  double min_duration{0.01};
};

struct ToppProfile {
  std::vector<double> durations;
  std::vector<Eigen::Vector3d> velocities;
  std::vector<Eigen::Vector3d> accelerations;
  std::vector<double> yaw_rates;
  std::vector<double> yaw_accelerations;
  bool valid{false};
};

// Path-constrained TOPP-RA style parameterization on a polyline with a
// separate yaw timeline. Returns false when the path is degenerate; the
// caller must then keep the legacy smoothstep seed.
bool computeToppSeed(const std::vector<Eigen::Vector3d>& waypoints,
                     const std::vector<double>& unwrapped_yaws,
                     const Eigen::Vector3d& start_velocity,
                     const Eigen::Vector3d& start_acceleration,
                     const Eigen::Vector3d& goal_velocity,
                     const Eigen::Vector3d& goal_acceleration,
                     const ToppLimits& limits, ToppProfile* profile);

}  // namespace mighty
