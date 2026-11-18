#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mighty/frontend/types.hpp"

namespace mighty::frontend {

enum class EdgeValidationMode : std::uint8_t {
  kSearch,
  kCertified,
};

class QueryContext {
 public:
  class Cache;

  QueryContext(CancelPredicate cancelled = {},
               std::optional<std::chrono::steady_clock::time_point> deadline =
                   std::nullopt);
  ~QueryContext();
  QueryContext(const QueryContext&) = delete;
  QueryContext& operator=(const QueryContext&) = delete;
  QueryContext(QueryContext&&) noexcept;
  QueryContext& operator=(QueryContext&&) noexcept;

  bool stopRequested() const;
  bool cancellationSeen() const { return cancellation_seen_; }
  bool timeoutSeen() const { return timeout_seen_; }
  const CancelPredicate& cancelled() const { return cancelled_; }
  const std::optional<std::chrono::steady_clock::time_point>& deadline() const {
    return deadline_;
  }
  void setDeadline(
      std::optional<std::chrono::steady_clock::time_point> deadline) {
    deadline_ = std::move(deadline);
  }
  void setCancellationSeen() const { cancellation_seen_ = true; }
  void setTimeoutSeen() const { timeout_seen_ = true; }
  Cache& cache() const;

  mutable std::size_t collision_queries{0};
  mutable std::size_t collision_samples{0};
  mutable std::size_t edge_checks{0};
  mutable std::size_t valid_states{0};
  mutable std::size_t pose_cache_hits{0};
  mutable std::size_t fcl_queries{0};
  mutable std::size_t swept_aabb_candidates{0};
  mutable std::size_t swept_sat_calls{0};

 private:
  mutable bool cancellation_seen_{false};
  mutable bool timeout_seen_{false};
  CancelPredicate cancelled_;
  std::optional<std::chrono::steady_clock::time_point> deadline_;
  mutable std::unique_ptr<Cache> cache_;
};

class WholeBodyCollisionScene {
 public:
  WholeBodyCollisionScene(const CollisionConfig& config,
                          const std::string& map_polymap_path,
                          const std::string& body_polymap_path);
  ~WholeBodyCollisionScene();

  WholeBodyCollisionScene(WholeBodyCollisionScene&&) noexcept;
  WholeBodyCollisionScene& operator=(WholeBodyCollisionScene&&) noexcept;
  WholeBodyCollisionScene(const WholeBodyCollisionScene&) = delete;
  WholeBodyCollisionScene& operator=(const WholeBodyCollisionScene&) = delete;

  const ModelInfo& modelInfo() const;
  double bodyRadius() const;

  bool isPoseCollisionFree(const PoseState& state,
                           QueryContext* context = nullptr) const;
  bool isEdgeCollisionFree(const PoseState& start, const PoseState& finish,
                           QueryContext& context,
                           EdgeValidationMode mode) const;
  bool isCertifiedEdgeCollisionFree(const PoseState& start,
                                    const PoseState& finish) const;

  std::optional<double> nearestCollisionFreeYaw(
      const Eigen::Vector3d& position, double hinted_yaw,
      double reference_yaw) const;

  std::vector<PoseState> simplifyPath(const std::vector<PoseState>& raw,
                                      QueryContext& context) const;
  std::vector<PoseState> cleanPosePath(
      const std::vector<PoseState>& input) const;
  std::vector<PoseState> smoothPath(const std::vector<PoseState>& shortcut,
                                    QueryContext& context,
                                    std::size_t& accepted_updates) const;
  bool certifyPath(const std::vector<PoseState>& waypoints,
                   QueryContext& context) const;
  double pathCost(const std::vector<PoseState>& waypoints) const;
  std::vector<PoseState> postprocessPath(const std::vector<PoseState>& raw,
                                         QueryContext& context,
                                         PlanStatistics* statistics) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mighty::frontend
