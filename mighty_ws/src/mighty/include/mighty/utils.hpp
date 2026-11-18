#pragma once

#include <cmath>

#include <Eigen/Geometry>
#include <geometry_msgs/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace mighty_utils {

inline void quaternion2Euler(const tf2::Quaternion& quaternion, double& roll,
                             double& pitch, double& yaw) {
  tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
}

inline void quaternion2Euler(const Eigen::Quaterniond& quaternion, double& roll,
                             double& pitch, double& yaw) {
  quaternion2Euler(
      tf2::Quaternion(quaternion.x(), quaternion.y(), quaternion.z(),
                      quaternion.w()),
      roll, pitch, yaw);
}

inline void quaternion2Euler(const geometry_msgs::Quaternion& quaternion,
                             double& roll, double& pitch, double& yaw) {
  quaternion2Euler(
      tf2::Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w),
      roll, pitch, yaw);
}

inline void angle_wrap(double& angle) {
  constexpr double kPi = 3.14159265358979323846;
  angle = std::fmod(angle + kPi, 2.0 * kPi);
  if (angle < 0.0) angle += 2.0 * kPi;
  angle -= kPi;
}

}  // namespace mighty_utils
