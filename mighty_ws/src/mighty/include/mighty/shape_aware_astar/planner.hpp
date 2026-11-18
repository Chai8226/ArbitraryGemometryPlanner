#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "mighty/frontend/types.hpp"

namespace mighty::shape_aware_astar {

using PlannerConfig = frontend::FrontendConfig;
using PoseState = frontend::PoseState;
using GoalYawConstraint = frontend::GoalYawConstraint;
using Goal = frontend::Goal;
using PlanRequest = frontend::PlanRequest;
using PlanStatus = frontend::PlanStatus;
using PhaseStatistics = frontend::PhaseStatistics;
using PlanStatistics = frontend::PlanStatistics;
using PlanResult = frontend::PlanResult;
using ModelInfo = frontend::ModelInfo;

class ShapeAwareAstarPlanner {
 public:
  using CancelPredicate = frontend::CancelPredicate;

  ShapeAwareAstarPlanner(const PlannerConfig& config,
                         const std::string& map_polymap_path,
                         const std::string& body_polymap_path);
  ~ShapeAwareAstarPlanner();

  ShapeAwareAstarPlanner(ShapeAwareAstarPlanner&&) noexcept;
  ShapeAwareAstarPlanner& operator=(ShapeAwareAstarPlanner&&) noexcept;
  ShapeAwareAstarPlanner(const ShapeAwareAstarPlanner&) = delete;
  ShapeAwareAstarPlanner& operator=(const ShapeAwareAstarPlanner&) = delete;

  const ModelInfo& modelInfo() const;
  PlanResult plan(const PlanRequest& request,
                  CancelPredicate cancelled = {}) const;
  bool isPoseCollisionFree(const PoseState& state) const;
  bool isEdgeCollisionFree(const PoseState& start,
                           const PoseState& finish) const;
  std::optional<double> nearestCollisionFreeYaw(
      const Eigen::Vector3d& position, double hinted_yaw,
      double reference_yaw) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mighty::shape_aware_astar
