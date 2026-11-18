/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <atomic>
#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <pcl_conversions/pcl_conversions.h>
#include <dynus_interfaces/ComputationTimes.h>
#include <dynus_interfaces/GlobalPlanStatus.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Path.h>
#include <dynus_interfaces/SpeedyPath.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt8.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <mighty/utils.hpp>
#include <mighty/global_once_goal_gate.hpp>

#include "mighty/visualization_utils.hpp"
#include "mighty/mighty.hpp"
#include "mighty/mighty_type.hpp"

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <mighty/ros_parameters.hpp>
#include "tf2_ros/transform_broadcaster.h"

#include "dynus_interfaces/Goal.h"
#include "dynus_interfaces/State.h"
#include "dynus_interfaces/Trajectory.h"
#include "dynus_interfaces/YawOutput.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored \
    "-Wdeprecated-declarations"  // pcl::SAC_SAMPLE_SIZE is protected since PCL 1.8.0 // NOLINT
#include <pcl/sample_consensus/model_types.h>
#pragma GCC diagnostic pop
#include <algorithm>
#include <execution>
#include <memory>
#include <string>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <sensor_msgs/point_cloud2_iterator.h>

#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/ColorRGBA.h"

namespace mighty {

using PCLPoint = pcl::PointXYZ;
using PCLPointCloud = pcl::PointCloud<pcl::PointXYZ>;

struct BackendBenchmarkSample {
  GlobalPlanResult plan_result;
  std::vector<state> optimized_setpoints;
  double total_pipeline_time_ms{0.0};
  whole_body_safety::SafetyStatistics safety_statistics;
  double validated_minimum_beta{std::numeric_limits<double>::infinity()};
};

/** @brief Main ROS1 node for the MIGHTY trajectory planner.
 *
 *  Manages known-map state and goal subscriptions, trajectory publishing, and
 *  visualization. Orchestrates
 *  the Shape-aware A* frontend and the trajectory optimizer.
 */
class MIGHTY_NODE {
 public:
  /** @brief Construct the node, declare parameters, set up publishers/subscribers and timers. */
  explicit MIGHTY_NODE();
  void start() { planning_spinner_.start(); }

  /** @brief Destructor. Logs benchmarking data if enabled. */
  ~MIGHTY_NODE();

  BackendBenchmarkSample runBackendBenchmark(
      const state& start, const state& goal, double request_time);
  BackendBenchmarkSample runCertifiedPathBenchmark(
      const state& start,
      const std::vector<mighty::frontend::PoseState>& shape_path,
      double request_time);

 private:
  bool isGlobalOnceMode() const { return par_.planning_mode == "global_once"; }

  // Callbacks
  void stateCallback(const dynus_interfaces::State::ConstPtr msg);
  void terminalGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void terminalGoalCallbackImpl(const geometry_msgs::PoseStamped& msg);
  void goalReachedCheckCallback();

  // Others
  void setParameters();
  void printParameters();
  void createMarkerArrayFromVec_Vec3f(const vec_Vec3f& occupied_cells,
                                      const std_msgs::ColorRGBA& color, int namespace_id,
                                      double scale,
                                      visualization_msgs::MarkerArray* marker_array);
  void clearMarkerArray(
      visualization_msgs::MarkerArray& path_marker,
      ros::Publisher publisher);
  void runSim();
  void printComputationTime(bool result);
  void setComputationTimesToZero();
  void retrieveData();
  void retrieveDataForVisualizationAndTopics();
  void publishKnownPolyMap();
  void initializeShapeAwareAstar();
  void cancelPendingShapeAwareAstar();
  void publishShapeAwareAstarResult(const frontend::PlanResult& result,
                          std::uint64_t generation);
  void clearShapeAwareAstarPlanVisualization(std::uint64_t generation);
  void publishShapeAwareAstarStatus(const std::string& status,
                          std::uint64_t generation);
  void publishShapeAwareAstarCurrentBody(
      const dynus_interfaces::State& measured_state);

  // Functions to publish
  void publishTraj();
  void publishActualTraj();
  void publishGoal();          // Publish the goal (trajectory points)
  void publishTrajectory();    // Publish the full trajectory with replan support
  void publishMpcPath();       // Publish smoothed global path as nav_msgs/Path for MPC
  void publishPointG() const;  // Publish the point G (projected terminal goal)
  void publishPointE() const;  // Publish the trajectory endpoint E
  void publishPointA() const;  // Publish the point A (starting point)
  void publishCurrentState(const state& state) const;
  void publishState(
      const state& data,
      const ros::Publisher& publisher) const;
  void publishCps();
  void publishStaticPushPoints();
  void publishVelocityInText(const Eigen::Vector3d& position, double velocity);
  void publishFsmStatus();
  void publishComputationTimes(double total_ms);
  void publishMiddleStageTime(const GlobalPlanResult& result);
  void publishGlobalPlanStatus(const GlobalPlanResult& result);
  void publishGlobalPlanStatus(bool success);
  // Timers for callback
  ros::WallTimer timer_goal_;
  ros::WallTimer timer_goal_reached_check_;
  ros::WallTimer timer_fsm_status_;

  // Queue members outlive handles and spinner.
  ros::CallbackQueue planning_queue_;
  ros::NodeHandle nh_, private_nh_, planning_nh_;
  ros::AsyncSpinner planning_spinner_{1, &planning_queue_};

  // Publishers
  ros::Publisher pub_known_map_;
  ros::Publisher pub_known_polymap_;
  ros::Publisher
      pub_polymap_boundary_;
  ros::Publisher pub_traj_committed_colored_;
  ros::Publisher pub_traj_middle_;
  ros::Publisher pub_traj_backend_;
  ros::Publisher pub_goal_;
  ros::Publisher pub_trajectory_;
  ros::Publisher pub_mpc_path_;
  ros::Publisher pub_point_G_;
  ros::Publisher pub_point_E_;
  ros::Publisher pub_point_G_term_;
  ros::Publisher pub_point_A_;
  ros::Publisher pub_current_state_;
  ros::Publisher pub_goal_reached_;
  ros::Publisher pub_fsm_status_;
  ros::Publisher
      pub_computation_times_;
  ros::Publisher pub_middle_stage_time_;
  ros::Publisher
      pub_global_plan_status_;
  ros::Publisher pub_setpoint_;
  ros::Publisher pub_actual_traj_;
  ros::Publisher pub_yaw_output_;
  ros::Publisher pub_cp_;
  ros::Publisher pub_static_push_points_;
  ros::Publisher pub_p_points_;
  ros::Publisher pub_vel_text_;
  // Shape-aware Weighted A* frontend outputs. These topics are visualization/status
  // only and are never consumed by MIGHTY's original backend or executor.
  ros::Publisher pub_shape_aware_astar_path_;
  ros::Publisher
      pub_shape_aware_astar_ghosts_;
  ros::Publisher
      pub_shape_aware_astar_current_body_;
  ros::Publisher pub_shape_aware_astar_status_;

  // Subscribers
  ros::Subscriber sub_state_;
  ros::Subscriber sub_terminal_goal_;

  // Wall-clock seconds of the last visualization publish in replanCallback.
  // The replan loop runs at 100 Hz which is fine for control but floods RViz
  // (especially for traj_committed_colored and the other
  // multi-KB MarkerArrays). RViz drops messages when the publish rate exceeds
  // its processing capacity, showing "some messages were lost" warnings.
  // We throttle the entire viz block to ~20 Hz (every 50 ms) — plenty smooth
  // visually, ~5x cheaper to render, no message loss.
  double last_replan_viz_publish_t_ = 0.0;

  // Same throttle for actual_traj. publishActualTraj() is called from
  // stateCallback which fires at the state publisher's rate (100 Hz from
  // fake_sim), and the persistent LINE_STRIP marker grows large enough that
  // RViz drops messages at that rate too.
  double last_actual_traj_publish_t_ = 0.0;

  // Visualization
  visualization_msgs::MarkerArray traj_committed_colored_;

  // Mutex
  std::mutex cloud_callback_mutex_;  // Mutex for cloud callback

  // Parameters
  int id_;
  std::string ns_;
  std::string id_str_;
  parameters par_;
  std::atomic<mighty::GlobalOncePhase> global_once_phase_{
      mighty::GlobalOncePhase::IDLE};
  uint32_t trajectory_id_ = 0;             // Trajectory ID for replan detection
  double final_g_ = 0.0;                   // only for debugging
  bool verbose_computation_time_ = false;  // only for debugging

  // DYNUS pointer
  std::shared_ptr<MIGHTY> mighty_ptr_;

  // Shape-aware Weighted A* visualization state. Planning is owned by MIGHTY.
  bool shape_aware_astar_available_{false};
  std::string shape_aware_astar_shape_{"t"};
  std::string known_map_id_;
  std::string known_map_shape_set_;
  // Wireframe of the selected body PLY (vertices + unique edges), rendered as
  // POINTS and LINE_LIST markers instead of the mesh resource.
  std::vector<Eigen::Vector3d> shape_aware_astar_body_vertices_;
  std::vector<std::pair<std::size_t, std::size_t>>
      shape_aware_astar_body_edges_;
  // Declared visualization.boundary_solids: full-coverage floor/ceiling
  // slabs are collision geometry but are excluded from the polymap surface
  // display and each is rendered as one cuboid on /polymap/boundary.
  struct BoundarySolid
  {
    std::string name;
    double z_min;
    double z_max;
  };
  std::vector<BoundarySolid> boundary_solids_;
  double shape_aware_astar_ghost_spacing_{1.0};
  std::size_t shape_aware_astar_max_ghosts_{100};
  frontend::FrontendConfig frontend_config_;
  std::atomic<std::uint64_t> shape_aware_astar_generation_{0};
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::string shape_body_frame_id_;
  std::atomic<bool> shape_aware_astar_current_body_latched_{false};
  // Keep map -> shape_body timestamps monotonic even for delayed samples.
  std::mutex shape_body_tf_mutex_;
  ros::Time shape_body_tf_last_stamp_{0};

  // debug
  double replan_last_time_called_ = 0.0;

  // Record computation time
  double global_planning_time_ = 0.0;
  double initial_guess_computation_time_ = 0.0;  // Time for computing initial guess
  double trajectory_optimization_time_ = 0.0;
  double replanning_computation_time_ = 0.0;
  double current_time_for_debug_ = 0.0;

  // Visualization
  std::vector<state> goal_setpoints_;

  // Yaw Optimization Debugging

  // Trajectory debugging
  std::vector<Eigen::Matrix<double, 3, 6>> cps_;

  // Static push points and p points
  vec_Vecf<3> static_push_points_;
  vec_Vecf<3> p_points_;
  int static_push_points_id_ = 0;
  int p_points_id_ = 0;

  // Published trajectory polynomial
  PieceWiseQuinticPol pwp_to_share_;

  // Flags
  bool state_initialized_ = false;           // State initialized

  // Actual-trajectory history for visualization (sando-style: a single
  // persistent LINE_STRIP marker built from a bounded history of past states,
  // colored by speed). Replaces the legacy "publish one ARROW marker per
  // sample" approach which leaked thousands of markers into RViz and lost
  // them frequently because the topic was a single Marker (not MarkerArray).
  std::vector<state> actual_traj_hist_;
  bool actual_traj_initialized_ = false;
  Eigen::Vector3d actual_traj_prev_pos_{0.0, 0.0, 0.0};
  double actual_traj_prev_time_ = 0.0;
  size_t actual_traj_max_hist_       = 4000;  // max stored states
  size_t actual_traj_max_points_vis_ = 300;   // max points shown in RViz after downsampling
  double actual_traj_line_width_     = 0.15;  // meters


  // D435 parameters
  std::string lidar_frame_id_;
  std::string d435_camera_info_topic_;

  // Timer to make sure we don't sample point cloud too often
  ros::Time last_lidar_callback_time_;
  ros::Time last_depth_camera_callback_time_;

  // Command-to-execution timing (time from goal received to first trajectory)
  ros::Time goal_received_time_;
  bool waiting_for_first_traj_ = false;
  ros::Publisher pub_command_to_exec_time_;
};

}  // namespace mighty
