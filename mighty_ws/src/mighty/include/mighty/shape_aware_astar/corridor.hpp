#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace mighty::shape_aware_astar {

inline std::uint64_t packGridCell(int x, int y, int z) {
  const auto encode = [](int value) -> std::uint64_t {
    return static_cast<std::uint64_t>(static_cast<std::uint32_t>(value + (1 << 20)));
  };
  return encode(x) | (encode(y) << 21U) | (encode(z) << 42U);
}

inline bool ballContains(const Eigen::Vector3d& position,
                         const Eigen::Vector3d& center, double radius) {
  return (position - center).norm() <= radius;
}

inline bool inBallUnion(const Eigen::Vector3d& position,
                        const std::vector<Eigen::Vector3d>& centers,
                        double radius) {
  for (const auto& center : centers) {
    if (ballContains(position, center, radius)) return true;
  }
  return false;
}

inline std::unordered_set<std::uint64_t> rasterizeBallUnion(
    const std::vector<Eigen::Vector3d>& centers,
    const Eigen::Vector3d& origin, double resolution, double radius,
    const Eigen::Array3i& minimum_index, const Eigen::Array3i& maximum_index,
    double tolerance) {
  std::unordered_set<std::uint64_t> cells;
  if (centers.empty() || resolution <= 0.0 || radius < 0.0) return cells;
  const double limit = radius + tolerance;
  const int reach =
      static_cast<int>(std::ceil((radius + std::abs(tolerance)) / resolution)) + 1;
  for (const auto& center : centers) {
    const Eigen::Vector3d delta = (center - origin) / resolution;
    const int cx = static_cast<int>(std::lround(delta.x()));
    const int cy = static_cast<int>(std::lround(delta.y()));
    const int cz = static_cast<int>(std::lround(delta.z()));
    const int x0 = std::max(minimum_index.x(), cx - reach);
    const int x1 = std::min(maximum_index.x(), cx + reach);
    const int y0 = std::max(minimum_index.y(), cy - reach);
    const int y1 = std::min(maximum_index.y(), cy + reach);
    const int z0 = std::max(minimum_index.z(), cz - reach);
    const int z1 = std::min(maximum_index.z(), cz + reach);
    for (int x = x0; x <= x1; ++x) {
      for (int y = y0; y <= y1; ++y) {
        for (int z = z0; z <= z1; ++z) {
          const Eigen::Vector3d position =
              origin + resolution * Eigen::Vector3d(static_cast<double>(x),
                                                    static_cast<double>(y),
                                                    static_cast<double>(z));
          if ((position - center).norm() <= limit) {
            cells.insert(packGridCell(x, y, z));
          }
        }
      }
    }
  }
  return cells;
}

}  // namespace mighty::shape_aware_astar
