#include <mighty/whole_body_safety.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>

namespace safety = mighty::whole_body_safety;

safety::ConvexPolytope box(const Eigen::Vector3d& center,
                          const Eigen::Vector3d& half_extent) {
  safety::ConvexPolytope result;
  result.A.resize(6, 3);
  result.b.resize(6);
  for (int axis = 0; axis < 3; ++axis) {
    result.A.row(2 * axis) = Eigen::Vector3d::Unit(axis);
    result.A.row(2 * axis + 1) = -Eigen::Vector3d::Unit(axis);
    result.b[2 * axis] = center[axis] + half_extent[axis];
    result.b[2 * axis + 1] = -center[axis] + half_extent[axis];
  }
  for (int mask = 0; mask < 8; ++mask) {
    Eigen::Vector3d vertex = center;
    for (int axis = 0; axis < 3; ++axis) {
      vertex[axis] += (mask & (1 << axis) ? 1.0 : -1.0) * half_extent[axis];
    }
    result.vertices.push_back(vertex);
  }
  result.finalize();
  return result;
}

int main() {
  // Asymmetric, off-origin geometry exercises the body yaw derivative as
  // well as the shared LP value and its translation gradient.
  const auto body = box({0.13, -0.07, 0.04}, {0.45, 0.16, 0.12});
  const auto obstacle = box({1.0, 0.37, 0.11}, {0.23, 0.29, 0.31});
  safety::ScalingProximitySolver solver;
  double worst_error = 0.0;
  for (const double yaw : {-1.1, -0.43, 0.21, 0.68, 1.24}) {
    const Eigen::Vector3d position(0.03, -0.09, 0.02);
    const auto result = solver.solve(body, obstacle, position, yaw);
    if (!result.success) return 1;
    for (int axis = 0; axis < 4; ++axis) {
      constexpr double step = 1.0e-5;
      Eigen::Vector3d plus = position, minus = position;
      double plus_yaw = yaw, minus_yaw = yaw;
      if (axis < 3) {
        plus[axis] += step;
        minus[axis] -= step;
      } else {
        plus_yaw += step;
        minus_yaw -= step;
      }
      double plus_beta, minus_beta;
      if (!solver.minimumBeta(body, obstacle, plus, plus_yaw, plus_beta) ||
          !solver.minimumBeta(body, obstacle, minus, minus_yaw, minus_beta)) {
        return 1;
      }
      const double numeric = (plus_beta - minus_beta) / (2.0 * step);
      const double analytic = axis < 3 ? result.gradient_position[axis]
                                       : result.gradient_yaw;
      const double error = std::abs(numeric - analytic) /
                           std::max(1.0, std::abs(numeric));
      worst_error = std::max(worst_error, error);
      if (error > 1.0e-4) {
        std::cerr << "yaw=" << yaw << " axis=" << axis
                  << " analytic=" << analytic << " numeric=" << numeric << '\n';
        return 1;
      }
    }
  }
  const auto root = std::filesystem::absolute(__FILE__).parent_path().parent_path();
  const auto scene = std::make_shared<safety::SafetyScene>(safety::SafetyScene::load(
      (root / "maps/polymap/block.polymap.json").string(),
      (root / "shape/normal/ply/t.polymap.json").string(), 0.1, 3.5));
  safety::WholeBodySafety oracle(scene, {});
  auto optimization_context = oracle.createContext();
  int active_poses = 0;
  for (const double yaw : {-0.43, 0.21, 0.68, 1.24}) {
    const Eigen::Vector3d position(3.0743125, -0.71915, 1.2845);
    const auto optimized = oracle.evaluateForOptimization(
        position, yaw, optimization_context);
    const auto reference = oracle.evaluate(position, yaw);
    if (!optimized.success || !reference.success ||
        std::abs(optimized.cost - reference.cost) >
            1.0e-5 * std::max(1.0, reference.cost)) {
      std::cerr << "optimization cost differs from cold safety query\n";
      return 1;
    }
    if (optimized.cost > 0.0) ++active_poses;
    for (int axis = 0; axis < 4; ++axis) {
      constexpr double step = 1.0e-5;
      Eigen::Vector3d plus = position, minus = position;
      double plus_yaw = yaw, minus_yaw = yaw;
      if (axis < 3) { plus[axis] += step; minus[axis] -= step; }
      else { plus_yaw += step; minus_yaw -= step; }
      const auto fp = oracle.evaluate(plus, plus_yaw);
      const auto fm = oracle.evaluate(minus, minus_yaw);
      const double numeric = (fp.cost - fm.cost) / (2.0 * step);
      const double analytic = axis < 3 ? optimized.gradient_position[axis]
                                       : optimized.gradient_yaw;
      const double error = std::abs(numeric - analytic) /
                           std::max(1.0, std::abs(numeric));
      if (!fp.success || !fm.success || error > 1.0e-3) {
        std::cerr << "safety cost yaw=" << yaw << " axis=" << axis
                  << " analytic=" << analytic << " numeric=" << numeric << '\n';
        return 1;
      }
    }
  }
  if (active_poses == 0) return 1;
  // At this measured contact transition, a 1e-5 yaw stencil straddles
  // different active faces and reverses the local derivative.  The optimizer's
  // finer retry stencil must agree with independent cold value differences.
  const auto cuboid_scene = std::make_shared<safety::SafetyScene>(safety::SafetyScene::load(
      (root / "maps/polymap/block.polymap.json").string(),
      (root / "shape/normal/ply/cuboid.polymap.json").string(), 0.1, 3.5));
  safety::WholeBodySafety cuboid_oracle(cuboid_scene, {});
  auto fine_context = cuboid_oracle.createContext();
  fine_context.optimization_derivative_step = 1.0e-7;
  const Eigen::Vector3d contact_position(
      -0.093978171876276592, 2.4052924401082825, 1.0139519988225483);
  const double contact_yaw = 0.12238606463687655;
  const auto fine = cuboid_oracle.evaluateForOptimization(
      contact_position, contact_yaw, fine_context);
  if (!fine.success) return 1;
  for (const double step : {1.0e-7, 1.0e-8}) {
    const auto plus = cuboid_oracle.evaluate(contact_position, contact_yaw + step);
    const auto minus = cuboid_oracle.evaluate(contact_position, contact_yaw - step);
    const double numeric = (plus.cost - minus.cost) / (2.0 * step);
    const double error = std::abs(fine.gradient_yaw - numeric) /
                         std::max(1.0, std::abs(numeric));
    if (!plus.success || !minus.success || !std::isfinite(error) || error > 1.0e-3) {
      std::cerr << "contact-transition yaw gradient=" << fine.gradient_yaw
                << " cold difference=" << numeric << '\n';
      return 1;
    }
  }
  std::cout << "whole-body scaling gradient: max relative error="
            << worst_error << '\n';
}
