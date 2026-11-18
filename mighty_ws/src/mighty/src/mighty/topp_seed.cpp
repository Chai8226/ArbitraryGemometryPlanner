#include "mighty/topp_seed.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mighty {
namespace {

constexpr double kEps = 1.0e-9;

double clampAbs(double value, double limit) {
  if (!(limit > 0.0)) return 0.0;
  return std::clamp(value, -limit, limit);
}

}  // namespace

bool computeToppSeed(const std::vector<Eigen::Vector3d>& waypoints,
                     const std::vector<double>& unwrapped_yaws,
                     const Eigen::Vector3d& start_velocity,
                     const Eigen::Vector3d& start_acceleration,
                     const Eigen::Vector3d& goal_velocity,
                     const Eigen::Vector3d& goal_acceleration,
                     const ToppLimits& limits, ToppProfile* profile) {
  if (!profile) return false;
  *profile = {};
  const int knots = static_cast<int>(waypoints.size());
  if (knots < 2 || static_cast<int>(unwrapped_yaws.size()) != knots) {
    return false;
  }
  for (const auto& point : waypoints) {
    if (!point.allFinite()) return false;
  }

  const int segments = knots - 1;
  std::vector<double> length(segments, 0.0);
  std::vector<Eigen::Vector3d> tangent(segments, Eigen::Vector3d::UnitX());
  for (int i = 0; i < segments; ++i) {
    const Eigen::Vector3d delta = waypoints[i + 1] - waypoints[i];
    length[i] = delta.norm();
    if (length[i] > kEps) tangent[i] = delta / length[i];
  }

  std::vector<double> v_limit(knots, limits.v_max);
  v_limit.front() = std::min(limits.v_max, start_velocity.norm());
  v_limit.back() = std::min(limits.v_max, goal_velocity.norm());
  for (int i = 1; i < knots - 1; ++i) {
    const Eigen::Vector3d incoming =
        length[i - 1] > kEps ? tangent[i - 1] : Eigen::Vector3d::Zero();
    const Eigen::Vector3d outgoing =
        length[i] > kEps ? tangent[i] : Eigen::Vector3d::Zero();
    const double cos_turn = std::clamp(incoming.dot(outgoing), -1.0, 1.0);
    const double turn = std::acos(cos_turn);
    const double span = std::max(kEps, std::min(length[i - 1], length[i]));
    // Corner speed from a local centripetal bound (TOPP-RA / TOPPQuad style).
    const double corner =
        turn <= 1.0e-6
            ? limits.v_max
            : std::sqrt(std::max(0.0, limits.a_max * span / std::max(turn, kEps)));
    v_limit[i] = std::min(limits.v_max, corner);
  }

  std::vector<double> speed(v_limit);
  speed.back() = v_limit.back();
  for (int i = knots - 2; i >= 0; --i) {
    const double reachable =
        std::sqrt(std::max(0.0, speed[i + 1] * speed[i + 1] +
                                    2.0 * limits.a_max * length[i]));
    speed[i] = std::min(speed[i], reachable);
  }
  speed.front() = std::min(speed.front(), v_limit.front());
  for (int i = 0; i < segments; ++i) {
    const double reachable =
        std::sqrt(std::max(0.0, speed[i] * speed[i] +
                                    2.0 * limits.a_max * length[i]));
    speed[i + 1] = std::min(speed[i + 1], reachable);
  }

  std::vector<double> durations(segments, limits.min_duration);
  for (int i = 0; i < segments; ++i) {
    // Do not floor the mean speed to kEps: for a rest-to-rest segment both
    // endpoint speeds vanish and length/1e-9 would yield a ~1e9*length s
    // duration, which freezes the delivered trajectory.  Skip the term in
    // that case; the accel/jerk bounds below remain valid lower bounds.
    const double mean_speed = 0.5 * (speed[i] + speed[i + 1]);
    double t = limits.min_duration;
    if (mean_speed > kEps) {
      t = std::max(t, length[i] / mean_speed);
    }
    if (limits.a_max > kEps) {
      t = std::max(t, std::abs(speed[i + 1] - speed[i]) / limits.a_max);
      t = std::max(t, std::sqrt(std::max(0.0, length[i] / limits.a_max)));
      if (speed[i] <= kEps && speed[i + 1] <= kEps) {
        // Rest-to-rest: symmetric bang-bang lower bound, tighter than the
        // constant-acceleration sqrt term above.
        t = std::max(
            t, 2.0 * std::sqrt(std::max(0.0, length[i] / limits.a_max)));
      }
    }
    if (limits.j_max > kEps) {
      t = std::max(t, std::cbrt(std::max(0.0, 6.0 * length[i] / limits.j_max)));
    }
    const double yaw_travel = std::abs(unwrapped_yaws[i + 1] - unwrapped_yaws[i]);
    if (limits.w_max > kEps) {
      t = std::max(t, yaw_travel / limits.w_max);
    }
    if (limits.yaw_acc_max > kEps) {
      t = std::max(t, std::sqrt(std::max(0.0, yaw_travel / limits.yaw_acc_max)));
    }
    durations[i] = std::max(t, limits.min_duration);
  }

  profile->durations = std::move(durations);
  profile->velocities.assign(knots, Eigen::Vector3d::Zero());
  profile->accelerations.assign(knots, Eigen::Vector3d::Zero());
  profile->yaw_rates.assign(knots, 0.0);
  profile->yaw_accelerations.assign(knots, 0.0);
  profile->velocities.front() = start_velocity;
  profile->accelerations.front() = start_acceleration;
  profile->velocities.back() = goal_velocity;
  profile->accelerations.back() = goal_acceleration;

  for (int i = 1; i < knots - 1; ++i) {
    Eigen::Vector3d direction = tangent[i - 1] + tangent[i];
    if (direction.norm() <= kEps) direction = tangent[i];
    if (direction.norm() > kEps) direction.normalize();
    profile->velocities[i] = speed[i] * direction;
    const Eigen::Vector3d incoming_v = speed[i - 1] * tangent[i - 1];
    const double dt = std::max(kEps, profile->durations[i - 1]);
    profile->accelerations[i] =
        (profile->velocities[i] - incoming_v) / dt;
    const double accel_norm = profile->accelerations[i].norm();
    if (accel_norm > limits.a_max && accel_norm > kEps) {
      profile->accelerations[i] *= limits.a_max / accel_norm;
    }
  }

  for (int i = 1; i < knots - 1; ++i) {
    const double dt_in = std::max(kEps, profile->durations[i - 1]);
    const double dt_out = std::max(kEps, profile->durations[i]);
    const double yaw_in =
        (unwrapped_yaws[i] - unwrapped_yaws[i - 1]) / dt_in;
    const double yaw_out =
        (unwrapped_yaws[i + 1] - unwrapped_yaws[i]) / dt_out;
    profile->yaw_rates[i] = clampAbs(0.5 * (yaw_in + yaw_out), limits.w_max);
    profile->yaw_accelerations[i] =
        clampAbs((yaw_out - yaw_in) / std::max(kEps, 0.5 * (dt_in + dt_out)),
                 limits.yaw_acc_max);
  }

  profile->valid = true;
  return true;
}

}  // namespace mighty
