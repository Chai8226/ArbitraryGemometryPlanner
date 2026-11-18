/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <mighty/mighty_node.hpp>
#include <mighty/polymap/artifact_format.hpp>
#include <mighty/polymap/visualization.hpp>
#include <mighty/runtime_guards.hpp>

#include <nlohmann/json.hpp>
#include <geometry_msgs/TransformStamped.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_set>

using namespace std::chrono_literals;

namespace mighty {

// Debug log: writes every published /goal and every received /term_goal to a
// file so the dip-toward-origin bug can be analyzed offline. Set debug_log_file
// to "" in the YAML to disable.
namespace {
double& debug_log_t0() {
  static double t0 = 0.0;
  return t0;
}
std::ofstream& debug_log_stream() {
  static std::ofstream s;  // never closed; flushed on every write
  return s;
}
void debug_log_open(const std::string& path) {
  if (path.empty()) return;
  auto& s = debug_log_stream();
  if (!s.is_open()) {
    s.open(path, std::ios::out | std::ios::trunc);
    if (s.is_open()) {
      s << std::fixed << std::setprecision(4);
      s << "# t_rel_sec | event | px | py | pz | extra\n";
      s.flush();
    }
  }
}
double debug_log_t(double now_sec) {
  auto& t0 = debug_log_t0();
  if (t0 == 0.0) t0 = now_sec;
  return now_sec - t0;
}

const std::unordered_set<std::string>& supportedBodyShapes(
    const std::string& shape_set) {
  static const std::unordered_set<std::string> small{
      "cross", "cuboid", "long1", "long2", "t", "v"};
  static const std::unordered_set<std::string> normal{
      "cross", "cuboid", "t", "v"};
  static const std::unordered_set<std::string> real{
      "drone_c_30", "drone_c_70", "drone_j_30", "drone_j_70"};
  static const std::unordered_set<std::string> empty;
  if (shape_set == "small") return small;
  if (shape_set == "normal") return normal;
  if (shape_set == "real") return real;
  return empty;
}

const char* globalPlanOutcomeName(GlobalPlanOutcome outcome) {
  switch (outcome) {
    case GlobalPlanOutcome::NOMINAL: return "nominal";
    case GlobalPlanOutcome::SAFE_DEGRADED: return "safe_degraded";
    case GlobalPlanOutcome::FAILED: return "failed";
    case GlobalPlanOutcome::NO_ACTION: return "no_action";
  }
  return "failed";
}

const char* trajectoryRouteName(TrajectoryRoute route) {
  switch (route) {
    case TrajectoryRoute::WHOLE_BODY_REFINED: return "whole_body_refined";
    case TrajectoryRoute::NONE: return "none";
  }
  return "none";
}

std::uint8_t globalPlanOutcomeMessage(GlobalPlanOutcome outcome) {
  switch (outcome) {
    case GlobalPlanOutcome::NOMINAL:
      return dynus_interfaces::GlobalPlanStatus::NOMINAL;
    case GlobalPlanOutcome::SAFE_DEGRADED:
      return dynus_interfaces::GlobalPlanStatus::SAFE_DEGRADED;
    case GlobalPlanOutcome::FAILED:
      return dynus_interfaces::GlobalPlanStatus::FAILED;
    case GlobalPlanOutcome::NO_ACTION:
      return dynus_interfaces::GlobalPlanStatus::NO_ACTION;
  }
  return dynus_interfaces::GlobalPlanStatus::FAILED;
}

std::uint8_t globalPlanStageMessage(GlobalPlanStage stage) {
  switch (stage) {
    case GlobalPlanStage::SUCCESS:
      return dynus_interfaces::GlobalPlanStatus::STAGE_SUCCESS;
    case GlobalPlanStage::ALREADY_REACHED:
      return dynus_interfaces::GlobalPlanStatus::STAGE_ALREADY_REACHED;
    case GlobalPlanStage::NOT_READY:
      return dynus_interfaces::GlobalPlanStatus::STAGE_NOT_READY;
    case GlobalPlanStage::SHAPE_ASTAR_FAILED:
      return dynus_interfaces::GlobalPlanStatus::STAGE_SHAPE_ASTAR_FAILED;
    case GlobalPlanStage::OPTIMIZATION_FAILED:
      return dynus_interfaces::GlobalPlanStatus::STAGE_OPTIMIZATION_FAILED;
  }
  return dynus_interfaces::GlobalPlanStatus::STAGE_OPTIMIZATION_FAILED;
}

std::uint8_t trajectoryRouteMessage(TrajectoryRoute route) {
  switch (route) {
    case TrajectoryRoute::WHOLE_BODY_REFINED:
      return dynus_interfaces::GlobalPlanStatus::ROUTE_WHOLE_BODY_REFINED;
    case TrajectoryRoute::NONE:
      return dynus_interfaces::GlobalPlanStatus::ROUTE_NONE;
  }
  return dynus_interfaces::GlobalPlanStatus::ROUTE_NONE;
}
}  // namespace

// ----------------------------------------------------------------------------

/**
 * @brief Constructor
 */
MIGHTY_NODE::MIGHTY_NODE() : private_nh_("~") {
  planning_nh_.setCallbackQueue(&planning_queue_);
  // Get id from ns
  ns_ = nh_.getNamespace();
  ns_ = ns_.substr(ns_.find_last_of("/") + 1);
  // Online namespaces end in a two-digit agent id (for example NX01). The
  // headless benchmark intentionally has no namespace, so give it a stable
  // non-conflicting local id without changing parameter-file matching.
  if (ns_.size() >= 2 && std::isdigit(static_cast<unsigned char>(ns_[ns_.size() - 2])) &&
      std::isdigit(static_cast<unsigned char>(ns_[ns_.size() - 1]))) {
    id_str_ = ns_.substr(ns_.size() - 2);
    id_ = std::stoi(id_str_);
  } else {
    id_str_ = "01";
    id_ = 1;
  }

  // Declare, set, and print parameters
  this->setParameters();
  if (par_.planning_mode != "global_once" || par_.map_mode != "known_pcd" ||
      par_.sim_env != "fake_sim") {
    throw std::invalid_argument(
        "mighty supports only global_once + known_pcd + fake_sim");
  }
  this->printParameters();

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>();
  shape_body_frame_id_ = ns_ + "/shape_body";

  // Visulaization publishers
  pub_known_map_ = nh_.advertise<sensor_msgs::PointCloud2>("known_map_cloud", 1, true);
  pub_known_polymap_ = nh_.advertise<visualization_msgs::MarkerArray>("polymap", 1, true);
  pub_polymap_boundary_ =
      nh_.advertise<visualization_msgs::MarkerArray>("polymap/boundary", 1, true);
  pub_shape_aware_astar_path_ =
      nh_.advertise<nav_msgs::Path>("shape_aware_astar/path", 1, true);
  pub_shape_aware_astar_ghosts_ =
      nh_.advertise<visualization_msgs::MarkerArray>("shape_aware_astar/body_ghosts", 1, false);
  pub_shape_aware_astar_current_body_ =
      nh_.advertise<visualization_msgs::MarkerArray>("shape_aware_astar/current_body", 1, true);
  pub_shape_aware_astar_status_ =
      nh_.advertise<std_msgs::String>("shape_aware_astar/status", 1, true);
  pub_traj_middle_ = nh_.advertise<visualization_msgs::MarkerArray>("traj_middle", 1, true);
  pub_traj_backend_ = nh_.advertise<visualization_msgs::MarkerArray>("traj_backend", 1, true);
  pub_traj_committed_colored_ = nh_.advertise<visualization_msgs::MarkerArray>("traj_committed_colored", 1, false);
  pub_setpoint_ = nh_.advertise<geometry_msgs::PointStamped>("setpoint_vis", 1, false);
  pub_actual_traj_ =
      nh_.advertise<visualization_msgs::MarkerArray>("actual_traj", 1, false);
  pub_cp_ = nh_.advertise<visualization_msgs::MarkerArray>("cp", 1, false);
  pub_static_push_points_ = nh_.advertise<visualization_msgs::MarkerArray>("static_push_points", 1, false);
  pub_p_points_ = nh_.advertise<visualization_msgs::MarkerArray>("p_points", 1, false);
  pub_point_A_ = nh_.advertise<geometry_msgs::PointStamped>("point_A", 1, false);
  pub_point_G_ = nh_.advertise<geometry_msgs::PointStamped>("point_G", 1, false);
  pub_point_E_ = nh_.advertise<geometry_msgs::PointStamped>("point_E", 1, false);
  pub_point_G_term_ = nh_.advertise<geometry_msgs::PointStamped>("point_G_term", 1, false);
  pub_current_state_ = nh_.advertise<geometry_msgs::PointStamped>("point_current_state", 1, false);
  pub_vel_text_ = nh_.advertise<visualization_msgs::Marker>("vel_text", 1, false);
  // Debug publishers
  pub_yaw_output_ = nh_.advertise<dynus_interfaces::YawOutput>("yaw_output", 10, false);

  // Essential publishers
  pub_goal_ = nh_.advertise<dynus_interfaces::Goal>("goal", 10, false);
  pub_trajectory_ =
      nh_.advertise<dynus_interfaces::Trajectory>("trajectory", 10, false);
  pub_mpc_path_ = nh_.advertise<dynus_interfaces::SpeedyPath>("mpc_waypoints", 10, false);
  pub_goal_reached_ = nh_.advertise<std_msgs::Empty>("goal_reached", 10, false);
  pub_fsm_status_ =
      nh_.advertise<std_msgs::UInt8>("fsm_state", 1, true);
  pub_computation_times_ =
      nh_.advertise<dynus_interfaces::ComputationTimes>("computation_times", 10, false);
  pub_middle_stage_time_ =
      nh_.advertise<std_msgs::Float64>("middle_stage_time_ms", 1, true);
  pub_global_plan_status_ =
      nh_.advertise<dynus_interfaces::GlobalPlanStatus>("global_plan_status", 1, true);
  pub_command_to_exec_time_ =
      nh_.advertise<std_msgs::Float64>("command_to_exec_time", 10, false);

  // Planning can take seconds; keep state/control callbacks on their own queue.
  sub_state_ = nh_.subscribe("state", 10, &MIGHTY_NODE::stateCallback, this);
  sub_terminal_goal_ = planning_nh_.subscribe("term_goal", 10, &MIGHTY_NODE::terminalGoalCallback, this);
  timer_goal_ = nh_.createWallTimer(ros::WallDuration(par_.dc),
      [this](const ros::WallTimerEvent&) { publishGoal(); }, false, false);
  timer_goal_reached_check_ = nh_.createWallTimer(ros::WallDuration(0.1),
      [this](const ros::WallTimerEvent&) { goalReachedCheckCallback(); });
  timer_fsm_status_ = nh_.createWallTimer(ros::WallDuration(0.5),
      [this](const ros::WallTimerEvent&) { publishFsmStatus(); });

  // Initialize the DYNUS object
  const std::filesystem::path shape_directory =
      std::filesystem::path(
          ros::package::getPath("mighty")) /
      "shape" / known_map_shape_set_ / "ply";
  const std::filesystem::path body_polymap_path =
      shape_directory / (shape_aware_astar_shape_ + ".polymap.json");
  mighty_ptr_ = std::make_shared<MIGHTY>(
      par_, frontend_config_, par_.known_map_polymap_path,
      body_polymap_path.string());
  if (mighty_ptr_->shapeModelInfo().map_name != known_map_id_) {
    throw std::runtime_error(
        "known_map.id does not match loaded map PolyMap name: expected '" +
        known_map_id_ + "', got '" +
        mighty_ptr_->shapeModelInfo().map_name + "'");
  }

  if (par_.map_mode == "known_pcd") {
    pcl::PointCloud<pcl::PointXYZ>::Ptr loaded(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(par_.known_map_pcd_path, *loaded) < 0) {
      throw std::runtime_error("failed to load known-map PCD: " +
                               par_.known_map_pcd_path);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    filtered->reserve(loaded->size());
    size_t rejected_non_finite = 0;
    size_t rejected_outside = 0;
    for (const auto& point : loaded->points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) {
        ++rejected_non_finite;
        continue;
      }
      if (point.x < par_.x_min || point.x >= par_.x_max ||
          point.y < par_.y_min || point.y >= par_.y_max ||
          point.z < par_.z_min || point.z >= par_.z_max) {
        ++rejected_outside;
        continue;
      }
      filtered->push_back(point);
    }
    filtered->width = static_cast<uint32_t>(filtered->size());
    filtered->height = 1;
    filtered->is_dense = true;
    if (filtered->empty()) {
      throw std::runtime_error(
          "known-map PCD has no finite points inside configured bounds");
    }
    if (rejected_non_finite > 0 || rejected_outside > 0) {
      ROS_WARN(
                  "Known map discarded %zu non-finite and %zu out-of-bounds points",
                  rejected_non_finite, rejected_outside);
    }

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*filtered, msg);
    msg.header.frame_id = par_.map_frame_id;
    msg.header.stamp = ros::Time::now();
    pub_known_map_.publish(msg);
    if (!par_.known_map_polymap_path.empty()) {
      publishKnownPolyMap();
    }
    ROS_INFO(
                "Known map ready: %zu obstacle points, frame=%s, "
                "bounds=[%.3f %.3f %.3f]-[%.3f %.3f %.3f], resolution=%.3f",
                filtered->size(), par_.map_frame_id.c_str(), par_.x_min, par_.y_min,
                par_.z_min, par_.x_max, par_.y_max, par_.z_max,
                par_.known_map_resolution);
  }

  initializeShapeAwareAstar();

  ROS_INFO(
              "Known-map mode: online mapping and rolling replanning are disabled");


}

// ----------------------------------------------------------------------------

/**
 * @brief Destructor
 */
MIGHTY_NODE::~MIGHTY_NODE() {
  planning_spinner_.stop();
  cancelPendingShapeAwareAstar();
  mighty_ptr_.reset();
}

// ----------------------------------------------------------------------------

BackendBenchmarkSample MIGHTY_NODE::runBackendBenchmark(
    const state& start, const state& goal, double request_time) {
  mighty_ptr_->updateState(start);
  const auto started = std::chrono::steady_clock::now();
  BackendBenchmarkSample sample;
  sample.plan_result = mighty_ptr_->planGlobalOnce(goal, request_time);
  sample.total_pipeline_time_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  mighty_ptr_->retrieveGoalSetpoints(sample.optimized_setpoints);
  return sample;
}

BackendBenchmarkSample MIGHTY_NODE::runCertifiedPathBenchmark(
    const state& start,
    const std::vector<mighty::frontend::PoseState>& shape_path,
    double request_time) {
  mighty_ptr_->updateState(start);
  const auto started = std::chrono::steady_clock::now();
  BackendBenchmarkSample sample;
  sample.plan_result =
      mighty_ptr_->planCertifiedShapePath(shape_path, request_time);
  sample.total_pipeline_time_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  mighty_ptr_->retrieveGoalSetpoints(sample.optimized_setpoints);
  return sample;
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishKnownPolyMap() {
  std::ifstream input(par_.known_map_polymap_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "failed to open known-map PolyMap: " + par_.known_map_polymap_path);
  }

  // The authoritative artifact also contains H-representations and complete
  // source-voxel attribution. They are not needed for RViz and can dominate
  // memory for large maps, so discard those subtrees while parsing.
  const auto keep_visualization_fields =
      [](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
        if (event != nlohmann::json::parse_event_t::key) {
          return true;
        }
        const std::string key = parsed.get<std::string>();
        return key != "source_voxels" && key != "halfspaces" &&
               key != "ads" &&
               key != "faces" && key != "source_pcd" &&
               key != "pcd_to_frame" && key != "decomposition" &&
               key != "cluster_id" && key != "component_id" &&
               key != "hull_volume" && key != "voxel_volume" &&
               key != "concavity";
      };
  nlohmann::json root =
      nlohmann::json::parse(input, keep_visualization_fields);
  if (root.at("format") != "mighty.polymap" ||
      root.at("format_version").get<int>() !=
          mighty::polymap::kArtifactFormatVersion ||
      root.at("kind") != "map") {
    throw std::runtime_error("unsupported or non-map PolyMap artifact");
  }

  const std::string expected_name =
      std::filesystem::path(par_.known_map_pcd_path).stem().string();
  const std::string artifact_name = root.at("name").get<std::string>();
  const std::string artifact_frame = root.at("frame_id").get<std::string>();
  if (artifact_name != expected_name) {
    throw std::runtime_error(
        "PolyMap name mismatch: expected '" + expected_name + "', got '" +
        artifact_name + "'");
  }
  if (artifact_frame != par_.map_frame_id) {
    throw std::runtime_error(
        "PolyMap frame mismatch: expected '" + par_.map_frame_id + "', got '" +
        artifact_frame + "'");
  }

  const auto json_vec3 = [](const nlohmann::json& value,
                            const std::string& field) {
    if (!value.is_array() || value.size() != 3) {
      throw std::runtime_error(field + " must contain three numbers");
    }
    std::array<double, 3> result{
        value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
    if (!std::all_of(result.begin(), result.end(),
                     [](double component) { return std::isfinite(component); })) {
      throw std::runtime_error(field + " contains a non-finite value");
    }
    return result;
  };

  const auto& grid = root.at("grid");
  const double artifact_resolution = grid.at("resolution").get<double>();
  const auto artifact_min = json_vec3(grid.at("min"), "grid.min");
  const auto artifact_max = json_vec3(grid.at("max"), "grid.max");
  constexpr double tolerance = 1e-6;
  const std::array<double, 3> configured_min{par_.x_min, par_.y_min, par_.z_min};
  const std::array<double, 3> configured_max{par_.x_max, par_.y_max, par_.z_max};
  if (!std::isfinite(artifact_resolution) ||
      std::abs(artifact_resolution - par_.known_map_resolution) > tolerance) {
    throw std::runtime_error("PolyMap resolution does not match known-map resolution");
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    if (std::abs(artifact_min[axis] - configured_min[axis]) > tolerance ||
        std::abs(artifact_max[axis] - configured_max[axis]) > tolerance) {
      throw std::runtime_error("PolyMap bounds do not match known-map bounds");
    }
  }

  const auto& polyhedra = root.at("polyhedra");
  if (!polyhedra.is_array() ||
      root.at("summary").at("polyhedra").get<size_t>() != polyhedra.size()) {
    throw std::runtime_error("PolyMap polyhedron count is inconsistent");
  }

  visualization_msgs::Marker clear;
  clear.header.frame_id = artifact_frame;
  clear.header.stamp = ros::Time::now();
  clear.action = visualization_msgs::Marker::DELETEALL;
  visualization_msgs::MarkerArray surface_snapshot, boundary_snapshot;
  surface_snapshot.markers.push_back(clear);

  constexpr size_t max_triangles_per_marker = 5000;
  const auto make_marker = [this, &artifact_frame](int id) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = artifact_frame;
    marker.header.stamp = ros::Time::now();
    marker.ns = "surface";
    marker.id = id;
    marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.r = 1.0F;
    marker.color.g = 1.0F;
    marker.color.b = 1.0F;
    marker.color.a = 0.68F;
    marker.points.reserve(max_triangles_per_marker * 3U);
    marker.colors.reserve(max_triangles_per_marker * 3U);
    return marker;
  };

  int marker_id = 0;
  size_t triangle_count = 0;
  visualization_msgs::Marker marker = make_marker(marker_id);
  std::vector<size_t> boundary_matches(boundary_solids_.size(), 0);
  for (size_t polyhedron_id = 0; polyhedron_id < polyhedra.size();
       ++polyhedron_id) {
    const auto& polyhedron = polyhedra[polyhedron_id];
    const auto rgb =
        mighty::polymap::distinctVisualizationColor(polyhedron_id);
    std_msgs::ColorRGBA color;
    color.r = rgb[0];
    color.g = rgb[1];
    color.b = rgb[2];
    color.a = 0.68F;
    const auto& json_vertices = polyhedron.at("vertices");
    const auto& json_triangles = polyhedron.at("triangles");
    if (!json_vertices.is_array() || !json_triangles.is_array()) {
      throw std::runtime_error("PolyMap vertices and triangles must be arrays");
    }
    std::vector<geometry_msgs::Point> vertices;
    vertices.reserve(json_vertices.size());
    for (const auto& json_vertex : json_vertices) {
      const auto vertex = json_vec3(json_vertex, "vertices[]");
      geometry_msgs::Point point;
      point.x = vertex[0];
      point.y = vertex[1];
      point.z = vertex[2];
      vertices.push_back(point);
    }
    // A declared floor/ceiling band claims a polyhedron iff the whole hull
    // lies inside it.  The planar-wall decomposition keeps slab pieces band
    // pure, so this removes the slabs from the surface display while walls
    // (which span the free band) are never affected.
    double hull_z_min = std::numeric_limits<double>::infinity();
    double hull_z_max = -std::numeric_limits<double>::infinity();
    for (const auto& point : vertices) {
      hull_z_min = std::min(hull_z_min, point.z);
      hull_z_max = std::max(hull_z_max, point.z);
    }
    size_t boundary_index = boundary_solids_.size();
    constexpr double kBoundaryTolerance = 1e-6;
    for (size_t band = 0; band < boundary_solids_.size(); ++band) {
      if (hull_z_min >= boundary_solids_[band].z_min - kBoundaryTolerance &&
          hull_z_max <= boundary_solids_[band].z_max + kBoundaryTolerance) {
        boundary_index = band;
        break;
      }
    }
    if (boundary_index != boundary_solids_.size()) {
      ++boundary_matches[boundary_index];
      continue;
    }
    for (const auto& triangle : json_triangles) {
      if (!triangle.is_array() || triangle.size() != 3) {
        throw std::runtime_error("triangles[] must contain three indices");
      }
      for (size_t corner = 0; corner < 3; ++corner) {
        const int index = triangle[corner].get<int>();
        if (index < 0 || static_cast<size_t>(index) >= vertices.size()) {
          throw std::runtime_error("PolyMap triangle index is out of range");
        }
        marker.points.push_back(vertices[static_cast<size_t>(index)]);
        marker.colors.push_back(color);
      }
      ++triangle_count;
      if (marker.points.size() / 3U >= max_triangles_per_marker) {
        surface_snapshot.markers.push_back(marker);
        marker = make_marker(++marker_id);
      }
    }
  }
  if (!marker.points.empty()) {
    surface_snapshot.markers.push_back(marker);
    ++marker_id;
  }

  size_t boundary_polyhedra = 0;
  if (!boundary_solids_.empty()) {
    visualization_msgs::Marker boundary_clear;
    boundary_clear.header.frame_id = artifact_frame;
    boundary_clear.header.stamp = ros::Time::now();
    boundary_clear.action = visualization_msgs::Marker::DELETEALL;
    boundary_snapshot.markers.push_back(boundary_clear);
  }
  for (size_t band = 0; band < boundary_solids_.size(); ++band) {
    const BoundarySolid& solid = boundary_solids_[band];
    if (boundary_matches[band] == 0) {
      ROS_WARN(
          "boundary solid '%s' matched no PolyMap polyhedron; nothing "
          "rendered on /polymap/boundary",
          solid.name.c_str());
      continue;
    }
    boundary_polyhedra += boundary_matches[band];
    visualization_msgs::Marker box;
    box.header.frame_id = artifact_frame;
    box.header.stamp = ros::Time::now();
    box.ns = "boundary";
    box.id = static_cast<int>(band);
    box.type = visualization_msgs::Marker::CUBE;
    box.action = visualization_msgs::Marker::ADD;
    box.pose.position.x = (artifact_min[0] + artifact_max[0]) / 2.0;
    box.pose.position.y = (artifact_min[1] + artifact_max[1]) / 2.0;
    box.pose.position.z = (solid.z_min + solid.z_max) / 2.0;
    box.pose.orientation.w = 1.0;
    box.scale.x = artifact_max[0] - artifact_min[0];
    box.scale.y = artifact_max[1] - artifact_min[1];
    box.scale.z = solid.z_max - solid.z_min;
    box.color.r = 0.6F;
    box.color.g = 0.6F;
    box.color.b = 0.6F;
    box.color.a = 0.30F;
    boundary_snapshot.markers.push_back(box);
  }

  if (boundary_polyhedra == 0) {
    ROS_INFO(
                "Known PolyMap ready: %zu polyhedra, %zu triangles, "
                "%d retained chunks",
                polyhedra.size(), triangle_count, marker_id);
  } else {
    ROS_INFO(
                "Known PolyMap ready: %zu polyhedra, %zu triangles, "
                "%d retained chunks, %zu boundary polyhedra moved to "
                "/polymap/boundary",
                polyhedra.size(), triangle_count, marker_id,
                boundary_polyhedra);
  }
  pub_known_polymap_.publish(surface_snapshot);
  pub_polymap_boundary_.publish(boundary_snapshot);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishShapeAwareAstarStatus(const std::string& status,
                                     std::uint64_t generation) {
  if (!pub_shape_aware_astar_status_ ||
      generation != shape_aware_astar_generation_.load()) {
    return;
  }
  std_msgs::String message;
  message.data = status;
  pub_shape_aware_astar_status_.publish(message);
}

void MIGHTY_NODE::initializeShapeAwareAstar() {
  shape_aware_astar_available_ = false;
  const std::uint64_t generation = shape_aware_astar_generation_.load();
  if (par_.planning_mode != "global_once" || par_.map_mode != "known_pcd") {
    const std::string reason =
        "unavailable: Shape-aware Weighted A* requires global_once + known_pcd";
    ROS_WARN( "[Shape-aware Weighted A*] %s", reason.c_str());
    publishShapeAwareAstarStatus(reason, generation);
    return;
  }

  const auto& supported_shapes = supportedBodyShapes(known_map_shape_set_);
  if (supported_shapes.count(shape_aware_astar_shape_) == 0) {
    const std::string reason =
        "unavailable: unsupported shape_aware_astar.shape='" + shape_aware_astar_shape_ +
        "' (paired PolyMap JSON and PLY are required)";
    ROS_ERROR( "[Shape-aware Weighted A*] %s", reason.c_str());
    publishShapeAwareAstarStatus(reason, generation);
    return;
  }
  if (par_.known_map_polymap_path.empty()) {
    const std::string reason =
        "unavailable: known_map_polymap_path is empty";
    ROS_ERROR( "[Shape-aware Weighted A*] %s", reason.c_str());
    publishShapeAwareAstarStatus(reason, generation);
    return;
  }

  try {
    const std::filesystem::path package_share =
        ros::package::getPath("mighty");
    const std::filesystem::path shape_directory =
        package_share / "shape" / known_map_shape_set_ / "ply";
    const std::filesystem::path body_polymap =
        shape_directory / (shape_aware_astar_shape_ + ".polymap.json");
    const std::filesystem::path body_ply =
        shape_directory / (shape_aware_astar_shape_ + ".ply");
    if (!std::filesystem::is_regular_file(body_polymap) ||
        !std::filesystem::is_regular_file(body_ply)) {
      throw std::runtime_error(
          "selected shape requires paired files: " +
          body_polymap.string() + " and " + body_ply.string());
    }
    if (!std::filesystem::is_regular_file(
            par_.known_map_polymap_path)) {
      throw std::runtime_error(
          "environment PolyMap not found: " +
          par_.known_map_polymap_path);
    }
    if (!std::isfinite(shape_aware_astar_ghost_spacing_) ||
        shape_aware_astar_ghost_spacing_ <= 0.0 || shape_aware_astar_max_ghosts_ == 0) {
      throw std::runtime_error(
          "ghost_spacing and max_ghosts must be positive");
    }

    const auto& model = mighty_ptr_->shapeModelInfo();

    if (!loadBodyWireframe(body_ply, shape_aware_astar_body_vertices_,
                           shape_aware_astar_body_edges_)) {
      throw std::runtime_error("failed to parse body PLY: " + body_ply.string());
    }
    shape_aware_astar_available_ = true;
    shape_aware_astar_current_body_latched_.store(false);
    {
      std::lock_guard<std::mutex> lock(shape_body_tf_mutex_);
      shape_body_tf_last_stamp_ = ros::Time(0);
    }
    std::ostringstream status;
    status << "ready map_id=" << known_map_id_
           << " shape_set=" << known_map_shape_set_
           << " shape=" << model.body_name
           << " body_components=" << model.body_components
           << " map=" << model.map_name
           << " obstacle_components=" << model.obstacle_components
           << " search=exact_weighted_astar"
           << " yaw_cost_weight="
           << frontend_config_.collision.yaw_cost_weight;
    publishShapeAwareAstarStatus(status.str(), generation);
    ROS_INFO( "[frontend] %s",
                status.str().c_str());
  } catch (const std::exception& error) {
    const std::string reason =
        std::string("unavailable: ") + error.what();
    ROS_ERROR( "[frontend] %s", reason.c_str());
    publishShapeAwareAstarStatus(reason, generation);
  }
}

void MIGHTY_NODE::cancelPendingShapeAwareAstar() {
  shape_aware_astar_generation_.fetch_add(1);
  if (mighty_ptr_) mighty_ptr_->cancelPendingGlobalPlan();
}

void MIGHTY_NODE::clearShapeAwareAstarPlanVisualization(
    std::uint64_t generation) {
  if (generation != shape_aware_astar_generation_.load()) return;

  nav_msgs::Path path;
  path.header.frame_id = par_.map_frame_id;
  path.header.stamp = ros::Time::now();
  pub_shape_aware_astar_path_.publish(path);

  visualization_msgs::MarkerArray ghosts;
  visualization_msgs::Marker delete_ghosts;
  delete_ghosts.header.frame_id = par_.map_frame_id;
  delete_ghosts.header.stamp = ros::Time::now();
  delete_ghosts.action = visualization_msgs::Marker::DELETEALL;
  ghosts.markers.push_back(delete_ghosts);
  pub_shape_aware_astar_ghosts_.publish(ghosts);
}

void MIGHTY_NODE::publishShapeAwareAstarResult(
    const frontend::PlanResult& result, std::uint64_t generation) {
  if (generation != shape_aware_astar_generation_.load() ||
      !result.success()) {
    return;
  }
  const ros::Time stamp = ros::Time::now();
  const auto set_pose =
      [](const frontend::PoseState& state,
         geometry_msgs::Pose* pose) {
        pose->position.x = state.position.x();
        pose->position.y = state.position.y();
        pose->position.z = state.position.z();
        pose->orientation.x = 0.0;
        pose->orientation.y = 0.0;
        pose->orientation.z = std::sin(0.5 * state.yaw);
        pose->orientation.w = std::cos(0.5 * state.yaw);
      };

  nav_msgs::Path path;
  path.header.frame_id = par_.map_frame_id;
  path.header.stamp = stamp;
  path.poses.reserve(result.waypoints.size());
  for (const auto& waypoint : result.waypoints) {
    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    set_pose(waypoint, &pose.pose);
    path.poses.push_back(pose);
  }
  if (generation != shape_aware_astar_generation_.load()) return;
  pub_shape_aware_astar_path_.publish(path);

  // Ghosts follow the committed mid/backend setpoints the sim tracks, not the
  // A* polyline. Sampling A* while current_body flies /goal looks like two
  // drones on two trajectories.
  if (mighty_ptr_) {
    mighty_ptr_->retrieveGoalSetpoints(goal_setpoints_);
  }
  std::vector<frontend::PoseState> ghost_source;
  if (goal_setpoints_.size() >= 2) {
    ghost_source.reserve(goal_setpoints_.size());
    for (const auto& setpoint : goal_setpoints_) {
      ghost_source.push_back({setpoint.pos, setpoint.yaw});
    }
  } else {
    ghost_source = result.waypoints;
  }
  const std::vector<frontend::PoseState> selected_ghosts =
      sampleBodyGhostPoses(ghost_source, shape_aware_astar_ghost_spacing_,
                           shape_aware_astar_max_ghosts_);

  visualization_msgs::MarkerArray ghosts;
  visualization_msgs::Marker delete_ghosts;
  delete_ghosts.header.frame_id = path.header.frame_id;
  delete_ghosts.header.stamp = ros::Time(0);
  delete_ghosts.action = visualization_msgs::Marker::DELETEALL;
  ghosts.markers.push_back(delete_ghosts);
  const ros::Time ghost_stamp(0);
  int ghost_id = 0;
  for (const auto& ghost_state : selected_ghosts) {
    geometry_msgs::Pose pose;
    set_pose(ghost_state, &pose);
    appendWireframeMarkers(shape_aware_astar_body_vertices_,
                           shape_aware_astar_body_edges_,
                           "shape_aware_astar_body_ghosts",
                           ghost_id * 2, pose, shapeWireframeGhostColor(), 0.018,
                           ghost_stamp, path.header.frame_id, false, &ghosts);
    ++ghost_id;
  }
  if (generation != shape_aware_astar_generation_.load()) return;
  pub_shape_aware_astar_ghosts_.publish(ghosts);

  std::ostringstream status;
  status << "success waypoints=" << result.waypoints.size()
         << " expansions=" << result.statistics.expansions
         << " generated_edges=" << result.statistics.generated_edges
         << " edge_checks=" << result.statistics.edge_checks
         << " path_length=" << result.statistics.path_length
         << " yaw_travel=" << result.statistics.yaw_travel
         << " final_cost=" << result.statistics.final_cost
         << " elapsed=" << result.statistics.elapsed_seconds
         << " first_certified="
         << result.statistics.first_certified_solution_seconds
         << " collision_samples="
         << result.statistics.collision_samples
         << " cache_hits=" << result.statistics.cache_hits
         << " mode=exact_weighted_astar"
         << " yaw_cost_weight="
         << result.statistics.yaw_cost_weight
         << " message=" << result.message;
  publishShapeAwareAstarStatus(status.str(), generation);
}

void MIGHTY_NODE::publishShapeAwareAstarCurrentBody(
    const dynus_interfaces::State& measured_state) {
  if (!shape_aware_astar_available_ ||
      shape_aware_astar_body_vertices_.empty() || !tf_broadcaster_) {
    return;
  }
  const std::array<double, 4> quaternion{
      measured_state.quat.x, measured_state.quat.y,
      measured_state.quat.z, measured_state.quat.w};
  if (!std::all_of(quaternion.begin(), quaternion.end(),
                   [](double value) { return std::isfinite(value); })) {
    return;
  }
  const double norm =
      std::sqrt(measured_state.quat.x * measured_state.quat.x +
                measured_state.quat.y * measured_state.quat.y +
                measured_state.quat.z * measured_state.quat.z +
                measured_state.quat.w * measured_state.quat.w);
  if (norm <= 1e-12) return;

  geometry_msgs::Quaternion normalized;
  normalized.x = measured_state.quat.x / norm;
  normalized.y = measured_state.quat.y / norm;
  normalized.z = measured_state.quat.z / norm;
  normalized.w = measured_state.quat.w / norm;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  mighty_utils::quaternion2Euler(normalized, roll, pitch, yaw);

  // Use the measured timestamp and only broadcast newer samples, preventing
  // delayed state messages from making RViz interpolate the body backwards.
  ros::Time stamp(measured_state.header.stamp);
  if (stamp.toNSec() == 0) stamp = ros::Time::now();

  geometry_msgs::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = par_.map_frame_id;
  transform.child_frame_id = shape_body_frame_id_;
  transform.transform.translation.x = measured_state.pos.x;
  transform.transform.translation.y = measured_state.pos.y;
  transform.transform.translation.z = measured_state.pos.z;
  transform.transform.rotation.x = 0.0;
  transform.transform.rotation.y = 0.0;
  transform.transform.rotation.z = std::sin(0.5 * yaw);
  transform.transform.rotation.w = std::cos(0.5 * yaw);
  {
    std::lock_guard<std::mutex> lock(shape_body_tf_mutex_);
    if (stamp <= shape_body_tf_last_stamp_) return;
    shape_body_tf_last_stamp_ = stamp;
    tf_broadcaster_->sendTransform(transform);
  }

  // Geometry is latched once in the yaw-only body frame. Republishing the
  // MarkerArray at 20 Hz made RViz rebuild the wireframe every tick. The
  // markers are frame_locked so RViz re-resolves map -> shape_body every
  // render frame and the single latched message keeps following the drone.
  if (shape_aware_astar_current_body_latched_.exchange(true)) return;

  geometry_msgs::Pose pose;
  pose.orientation.w = 1.0;
  visualization_msgs::MarkerArray markers;
  const ros::Time marker_stamp(0);
  appendWireframeMarkers(shape_aware_astar_body_vertices_,
                         shape_aware_astar_body_edges_,
                         "shape_aware_astar_current_body", 0, pose,
                         shapeWireframeLiveBodyColor(), 0.045, marker_stamp,
                         shape_body_frame_id_, true, &markers);
  pub_shape_aware_astar_current_body_.publish(markers);
}

// ----------------------------------------------------------------------------

/**
 * @brief Declare the parameters
 */


// ----------------------------------------------------------------------------

/**
 * @brief Set the parameters
 */
void MIGHTY_NODE::setParameters() {
  // Set the parameters

  // Sim enviroment
  par_.sim_env = mighty::readParameter<std::string>(private_nh_, "sim_env", "fake_sim");
  par_.planning_mode = mighty::readParameter<std::string>(private_nh_, "planning_mode", "global_once");
  par_.map_mode = mighty::readParameter<std::string>(private_nh_, "map_mode", "known_pcd");
  par_.known_map_pcd_path =
      mighty::readParameter<std::string>(private_nh_, "known_map_pcd_path", "");
  par_.known_map_polymap_path =
      mighty::readParameter<std::string>(private_nh_, "known_map_polymap_path", "");
  par_.known_map_resolution =
      mighty::readParameter<double>(private_nh_, "known_map.resolution", 0.1);
  known_map_id_ = mighty::readParameter<std::string>(private_nh_, "known_map.id", "");
  known_map_shape_set_ =
      mighty::readParameter<std::string>(private_nh_, "known_map.shape_set", "");
  boundary_solids_.clear();
  const std::string boundary_solids_value =
      mighty::readParameter<std::string>(private_nh_, "known_map.boundary_solids", "");
  if (!boundary_solids_value.empty()) {
    // Encoding: name:z_min:z_max,name2:z_min:z_max (run_known_map.py emits
    // the exact same format, with identifiers restricted to [A-Za-z0-9_]).
    size_t token_begin = 0;
    while (token_begin <= boundary_solids_value.size()) {
      const size_t token_end = boundary_solids_value.find(',', token_begin);
      const std::string token = boundary_solids_value.substr(
          token_begin, token_end == std::string::npos
                           ? std::string::npos
                           : token_end - token_begin);
      const size_t first_colon = token.find(':');
      const size_t second_colon =
          first_colon == std::string::npos ? std::string::npos
                                           : token.find(':', first_colon + 1);
      BoundarySolid solid;
      solid.name = token.substr(0, first_colon);
      const bool valid =
          first_colon != std::string::npos && second_colon != std::string::npos &&
          token.find(':', second_colon + 1) == std::string::npos &&
          !solid.name.empty();
      if (!valid) {
        throw std::invalid_argument(
            "known_map.boundary_solids entry must be name:z_min:z_max: " +
            token);
      }
      try {
        solid.z_min = std::stod(token.substr(first_colon + 1,
                                             second_colon - first_colon - 1));
        solid.z_max = std::stod(token.substr(second_colon + 1));
      } catch (const std::exception&) {
        throw std::invalid_argument(
            "known_map.boundary_solids z values must be numeric: " + token);
      }
      if (!std::isfinite(solid.z_min) || !std::isfinite(solid.z_max) ||
          solid.z_max <= solid.z_min) {
        throw std::invalid_argument(
            "known_map.boundary_solids z must be finite and increasing: " +
            token);
      }
      boundary_solids_.push_back(std::move(solid));
      if (token_end == std::string::npos) {
        break;
      }
      token_begin = token_end + 1;
    }
  }
  par_.global_once_refinement_timeout_seconds =
      mighty::readParameter<double>(private_nh_, "global_once.refinement_timeout_seconds", 3.0);
  par_.global_once_hard_timeout_seconds =
      mighty::readParameter<double>(private_nh_, "global_once.hard_timeout_seconds", 5.0);
  par_.global_once_minimum_interior_speed =
      mighty::readParameter<double>(private_nh_, "global_once.minimum_interior_speed", 0.2);
  if (par_.map_mode != "known_pcd") {
    throw std::invalid_argument("map_mode must be 'known_pcd'");
  }
  if (par_.known_map_pcd_path.empty()) {
    throw std::invalid_argument(
        "known_map_pcd_path is required when map_mode=known_pcd");
  }
  if (!(par_.global_once_refinement_timeout_seconds > 0.0) ||
      !(par_.global_once_hard_timeout_seconds > 0.0) ||
      par_.global_once_refinement_timeout_seconds >
          par_.global_once_hard_timeout_seconds) {
    throw std::invalid_argument(
        "global_once timeouts must satisfy 0 < refinement <= hard");
  }
  if (!std::isfinite(par_.global_once_minimum_interior_speed) ||
      par_.global_once_minimum_interior_speed <= 0.0) {
    throw std::invalid_argument(
        "global_once.minimum_interior_speed must be positive");
  }
  if (par_.planning_mode != "global_once") {
    throw std::invalid_argument("planning_mode must be 'global_once'");
  }

  par_.map_frame_id = mighty::readParameter<std::string>(private_nh_, "map_frame_id", "map");

  // Visual level
  par_.visual_level = mighty::readParameter<int>(private_nh_, "visual_level", 1);

  // Global Planner parameters
  par_.x_min = mighty::readParameter<double>(private_nh_, "x_min", -100.0);
  par_.x_max = mighty::readParameter<double>(private_nh_, "x_max", 100.0);
  par_.y_min = mighty::readParameter<double>(private_nh_, "y_min", -100.0);
  par_.y_max = mighty::readParameter<double>(private_nh_, "y_max", 100.0);
  par_.z_min = mighty::readParameter<double>(private_nh_, "z_min", 0.0);
  par_.z_max = mighty::readParameter<double>(private_nh_, "z_max", 5.0);
  par_.max_dist_vertexes = mighty::readParameter<double>(private_nh_, "max_dist_vertexes", 5.0);

  par_.use_state_update = mighty::readParameter<bool>(private_nh_, "use_state_update", true);

  // Optimization parameters
  par_.dc = mighty::readParameter<double>(private_nh_, "dc", 0.01);
  par_.v_max = mighty::readParameter<double>(private_nh_, "v_max", 1.0);
  if (par_.global_once_minimum_interior_speed >= par_.v_max) {
    throw std::invalid_argument(
        "global_once.minimum_interior_speed must be below v_max");
  }
  par_.a_max = mighty::readParameter<double>(private_nh_, "a_max", 1.0);
  par_.j_max = mighty::readParameter<double>(private_nh_, "j_max", 1.0);
  par_.jerk_weight = mighty::readParameter<double>(private_nh_, "jerk_weight", 1.0);
  par_.yaw_jerk_weight =
      mighty::readParameter<double>(private_nh_, "yaw_jerk_weight", 0.1);
  par_.yaw_reference_weight =
      mighty::readParameter<double>(private_nh_, "yaw_reference_weight", 0.0);
  par_.time_weight = mighty::readParameter<double>(private_nh_, "time_weight", 1.0);
  par_.pos_anchor_weight = mighty::readParameter<double>(private_nh_, "pos_anchor_weight", 1.0);
  par_.stat_weight = mighty::readParameter<double>(private_nh_, "stat_weight", 1.0);
  par_.dyn_constr_bodyrate_weight = mighty::readParameter<double>(private_nh_, "dyn_constr_bodyrate_weight", 1.0);
  par_.dyn_constr_tilt_weight = mighty::readParameter<double>(private_nh_, "dyn_constr_tilt_weight", 1.0);
  par_.dyn_constr_thrust_weight = mighty::readParameter<double>(private_nh_, "dyn_constr_thrust_weight", 1.0);
  par_.dyn_constr_vel_weight = mighty::readParameter<double>(private_nh_, "dyn_constr_vel_weight", 1.0);
  par_.dyn_constr_acc_weight = mighty::readParameter<double>(private_nh_, "dyn_constr_acc_weight", 1.0);
  par_.dyn_constr_jerk_weight = mighty::readParameter<double>(private_nh_, "dyn_constr_jerk_weight", 1.0);
  par_.planner_Co = mighty::readParameter<double>(private_nh_, "planner_Co", 0.1);
  par_.planner_Co_normal =
      mighty::readParameter<double>(private_nh_, "planner_Co_normal", 0.1);
  par_.planner_Co_small =
      mighty::readParameter<double>(private_nh_, "planner_Co_small", 0.05);
  par_.planner_Co_real =
      mighty::readParameter<double>(private_nh_, "planner_Co_real", 0.05);
  par_.init_mode = mighty::readParameter<std::string>(private_nh_, "init_mode", "topp");
  par_.light_activation_ladder =
      mighty::readParameter<bool>(private_nh_, "light_activation_ladder", true);
  par_.safety_beta_soft = mighty::readParameter<double>(private_nh_, "safety_beta_soft", 1.15);
  par_.safety_exponential_gain =
      mighty::readParameter<double>(private_nh_, "safety_exponential_gain", 6.0);
  par_.safety_hard_weight_multiplier =
      mighty::readParameter<double>(private_nh_, "safety_hard_weight_multiplier", 20.0);
  par_.safety_position_sample_step =
      mighty::readParameter<double>(private_nh_, "safety_position_sample_step", 0.2);
  par_.safety_yaw_sample_step =
      mighty::readParameter<double>(private_nh_, "safety_yaw_sample_step", 15.0 * M_PI / 180.0);
  par_.safety_ipm_tolerance =
      mighty::readParameter<double>(private_nh_, "safety_ipm_tolerance", 1.0e-7);
  par_.safety_ipm_warm_iterations =
      mighty::readParameter<int>(private_nh_, "safety_ipm_warm_iterations", 12);
  par_.safety_ipm_cold_iterations =
      mighty::readParameter<int>(private_nh_, "safety_ipm_cold_iterations", 30);
  par_.safety_ipm_regularization =
      mighty::readParameter<double>(private_nh_, "safety_ipm_regularization", 1.0e-9);
  par_.safety_acceptance_beta =
      mighty::readParameter<double>(private_nh_, "safety_acceptance_beta", 1.0 - 1.0e-4);
  verbose_computation_time_ = mighty::readParameter<bool>(private_nh_, "verbose_computation_time", false);
  par_.drone_bbox = mighty::readParameter<std::vector<double>>(private_nh_, "drone_bbox", std::vector<double>{0.5, 0.5, 0.5});
  par_.drone_radius = par_.drone_bbox[0] / 2.0;
  par_.goal_radius = mighty::readParameter<double>(private_nh_, "goal_radius", 0.5);
  par_.spline_degree = mighty::readParameter<int>(private_nh_, "spline_degree", 5);

  // L-BFGS parameters
  par_.f_dec_coeff = mighty::readParameter<double>(private_nh_, "f_dec_coeff", 1e-2);
  par_.cautious_factor = mighty::readParameter<double>(private_nh_, "cautious_factor", 0.0);
  par_.past = mighty::readParameter<int>(private_nh_, "past", 5);
  par_.max_linesearch = mighty::readParameter<int>(private_nh_, "max_linesearch", 64);
  par_.max_iterations = mighty::readParameter<int>(private_nh_, "max_iterations", 30);
  par_.g_epsilon = mighty::readParameter<double>(private_nh_, "g_epsilon", 0.0);
  par_.delta = mighty::readParameter<double>(private_nh_, "delta", 1e-6);

  // Yaw-related parameters
  par_.alpha_filter_dyaw = mighty::readParameter<double>(private_nh_, "alpha_filter_dyaw", 0.8);
  par_.w_max = mighty::readParameter<double>(private_nh_, "w_max", 0.5);
  par_.yaw_acc_max = mighty::readParameter<double>(private_nh_, "yaw_acc_max", 2.0);
  par_.yaw_goal_tolerance =
      mighty::readParameter<double>(private_nh_, "yaw_goal_tolerance", 0.04);
  par_.yaw_spinning_threshold = mighty::readParameter<int>(private_nh_, "yaw_spinning_threshold", 10);
  par_.yaw_spinning_dyaw = mighty::readParameter<double>(private_nh_, "yaw_spinning_dyaw", 0.1);

  // Simulation env parameters
  par_.force_goal_z = mighty::readParameter<bool>(private_nh_, "force_goal_z", true);
  par_.default_goal_z = mighty::readParameter<double>(private_nh_, "default_goal_z", 2.5);

  if (par_.default_goal_z <= par_.z_min) {
    ROS_ERROR( "Default goal z is lower than the ground level");
  }

  if (par_.default_goal_z >= par_.z_max) {
    ROS_ERROR( "Default goal z is higher than the max level");
  }

  // Debug flags
  par_.debug_verbose = mighty::readParameter<bool>(private_nh_, "debug_verbose", false);
  {
    auto path = mighty::readParameter<std::string>(private_nh_, "debug_log_file", std::string(""));
    debug_log_open(path);
    if (!path.empty()) {
      ROS_INFO( "Debug log file: %s", path.c_str());
    }
  }

  par_.trajectory_downsample_points = mighty::readParameter<int>(private_nh_, "trajectory_downsample_points", 500);
  par_.mpc_path_spacing = mighty::readParameter<double>(private_nh_, "mpc_path_spacing", 0.05);


  // Mandatory Shape-aware Weighted A* frontend configuration.
  shape_aware_astar_shape_ = mighty::readParameter<std::string>(private_nh_, "shape_aware_astar.shape", "t");
  std::transform(shape_aware_astar_shape_.begin(), shape_aware_astar_shape_.end(),
                 shape_aware_astar_shape_.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  std::transform(known_map_id_.begin(), known_map_id_.end(),
                 known_map_id_.begin(), [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  std::transform(known_map_shape_set_.begin(), known_map_shape_set_.end(),
                 known_map_shape_set_.begin(), [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  const std::string required_shape_set =
      known_map_id_ == "hole" ? "small"
      : (known_map_id_ == "forest" || known_map_id_ == "narrow" ||
                 known_map_id_ == "block"
             ? "normal"
             : (known_map_id_ == "indoor1" || known_map_id_ == "indoor2"
                    ? "real"
                    : ""));
  if (required_shape_set.empty() || known_map_shape_set_ != required_shape_set) {
    throw std::invalid_argument(
        "known_map.id/known_map.shape_set must be hole/small, forest/normal, "
        "narrow/normal, block/normal, indoor1/real, or indoor2/real");
  }
  const auto& supported_shapes = supportedBodyShapes(known_map_shape_set_);
  if (supported_shapes.count(shape_aware_astar_shape_) == 0) {
    throw std::invalid_argument(
        "shape_aware_astar.shape='" + shape_aware_astar_shape_ +
        "' is not available in shape_set='" + known_map_shape_set_ + "'");
  }
  if (known_map_shape_set_ == "small") {
    par_.planner_Co = par_.planner_Co_small;
  } else if (known_map_shape_set_ == "real") {
    par_.planner_Co = par_.planner_Co_real;
  } else {
    par_.planner_Co = par_.planner_Co_normal;
  }
  shape_aware_astar_ghost_spacing_ =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.ghost_spacing", 1.0);
  const auto max_ghosts =
      mighty::readParameter<int>(private_nh_, "shape_aware_astar.max_ghosts", 100);
  shape_aware_astar_max_ghosts_ =
      max_ghosts > 0 ? static_cast<std::size_t>(max_ghosts) : 0U;

  frontend_config_.timeout_seconds =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.timeout_seconds", 5.0);
  frontend_config_.collision.spatial_resolution =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.spatial_resolution", 0.20);
  frontend_config_.collision.yaw_resolution =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.yaw_resolution", 0.17453292519943295);
  frontend_config_.collision.max_yaw_delta_per_step =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.max_yaw_delta_per_step", 1.5707963267948966);
  frontend_config_.collision.yaw_cost_weight =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.yaw_cost_weight", 0.20);
  frontend_config_.collision.body_clearance = par_.planner_Co;
  frontend_config_.collision.collision_sample_max_distance =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.collision_sample_max_distance", 0.20);
  frontend_config_.collision.collision_sample_max_yaw_step =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.collision_sample_max_yaw_step", 0.17453292519943295);
  frontend_config_.collision.smoothing_enabled =
      mighty::readParameter<bool>(private_nh_, "shape_aware_astar.smoothing.enabled", true);
  frontend_config_.collision.smoothing_iterations = static_cast<std::size_t>(
      mighty::readParameter<int>(private_nh_, "shape_aware_astar.smoothing.iterations", 4));
  frontend_config_.collision.smoothing_alpha =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.smoothing.alpha", 0.35);
  frontend_config_.collision.virtual_ceiling_height =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.virtual_ceiling_height", 4.5);
  frontend_config_.collision.numeric_tolerance =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.numeric_tolerance", 1e-8);
  frontend_config_.max_expansions = static_cast<std::size_t>(
      mighty::readParameter<int>(private_nh_, "shape_aware_astar.max_expansions", 250000));
  frontend_config_.heuristic_weight =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.heuristic_weight", 1.5);
  frontend_config_.max_yaw_candidates = static_cast<std::size_t>(
      mighty::readParameter<int>(private_nh_, "shape_aware_astar.max_yaw_candidates", 5));
  frontend_config_.coarse_spatial_resolution =
      mighty::readParameter<double>(private_nh_, "shape_aware_astar.coarse_spatial_resolution", 0.40);

  if (par_.planning_mode != "global_once") {
    throw std::invalid_argument(
        "Shape A* frontend supports only planning_mode=global_once");
  }
  {
    if (par_.map_mode != "known_pcd") {
      throw std::invalid_argument(
          "planning_mode=global_once requires map_mode=known_pcd");
    }
  }
  {
    if (par_.planning_mode != "global_once" || par_.map_mode != "known_pcd") {
      throw std::invalid_argument(
          "yaw optimization requires known_pcd + global_once");
    }
    if (!std::isfinite(par_.w_max) || par_.w_max <= 0.0 ||
        !std::isfinite(par_.yaw_acc_max) || par_.yaw_acc_max <= 0.0 ||
        !std::isfinite(par_.yaw_jerk_weight) ||
        par_.yaw_jerk_weight < 0.0 ||
        !std::isfinite(par_.yaw_reference_weight) ||
        par_.yaw_reference_weight < 0.0 ||
        !std::isfinite(par_.yaw_goal_tolerance) ||
        par_.yaw_goal_tolerance < 0.0) {
      throw std::invalid_argument("invalid yaw optimization parameters");
    }
  }
}

// ----------------------------------------------------------------------------

/**
 * @brief Print the parameters
 */
void MIGHTY_NODE::printParameters() {
  // Print the parameters

  // Sim enviroment
  ROS_INFO( "Sim Enviroment: %s", par_.sim_env.c_str());
  ROS_INFO( "Planning Mode: %s", par_.planning_mode.c_str());

  // Visual
  ROS_INFO( "Visual Level: %d", par_.visual_level);

  ROS_INFO( "Initial Guess Planner: Shape-aware Weighted A*");
  ROS_INFO( "X Min: %f", par_.x_min);
  ROS_INFO( "X Max: %f", par_.x_max);
  ROS_INFO( "Y Min: %f", par_.y_min);
  ROS_INFO( "Y Max: %f", par_.y_max);
  ROS_INFO( "Z Ground: %f", par_.z_min);
  ROS_INFO( "Z Max: %f", par_.z_max);
  ROS_INFO( "max_dist_vertexes: %f", par_.max_dist_vertexes);

  ROS_INFO( "Use State Update?: %d", par_.use_state_update);

  // Optimization parameters
  ROS_INFO( "DC: %f", par_.dc);
  ROS_INFO( "V Max: %f", par_.v_max);
  ROS_INFO( "A Max: %f", par_.a_max);
  ROS_INFO( "J Max: %f", par_.j_max);
  ROS_INFO( "Control Cost Weight: %f", par_.jerk_weight);
  ROS_INFO( "Yaw Jerk Weight: %f",
              par_.yaw_jerk_weight);
  ROS_INFO( "Yaw Reference Weight: %f",
              par_.yaw_reference_weight);
  ROS_INFO( "Time Weight: %f", par_.time_weight);
  ROS_INFO( "Position Anchor Weight: %f", par_.pos_anchor_weight);
  ROS_INFO( "Static Obstacle Weight: %f", par_.stat_weight);
  ROS_INFO( "Violation of Bodyrate Constr. Weight: %f",
              par_.dyn_constr_bodyrate_weight);
  ROS_INFO( "Violation of Tilt Constr. Weight: %f",
              par_.dyn_constr_tilt_weight);
  ROS_INFO( "Violation of Thrust Constr. Weight: %f",
              par_.dyn_constr_thrust_weight);
  ROS_INFO( "Violation of Vel Constr. Weight: %f",
              par_.dyn_constr_vel_weight);
  ROS_INFO( "Violation of Aeccel Constr. Weight: %f",
              par_.dyn_constr_acc_weight);
  ROS_INFO( "Violation of Jerk Constr. Weight: %f",
              par_.dyn_constr_jerk_weight);
  ROS_INFO( "Whole-body Clearance: %f", par_.planner_Co);
  ROS_INFO( "Verbose Computation Time: %d", verbose_computation_time_);
  ROS_INFO( "Drone Bbox: (%f, %f, %f)", par_.drone_bbox[0],
              par_.drone_bbox[1], par_.drone_bbox[2]);
  ROS_INFO( "Goal Radius: %f", par_.goal_radius);

  // L-BFGS parameters
  ROS_INFO( "f_dec_coeff: %f", par_.f_dec_coeff);
  ROS_INFO( "Cautious Factor: %f", par_.cautious_factor);
  ROS_INFO( "Past: %d", par_.past);
  ROS_INFO( "Max Linesearch: %d", par_.max_linesearch);
  ROS_INFO( "Max Iterations: %d", par_.max_iterations);
  ROS_INFO( "g_epsilon: %f", par_.g_epsilon);
  ROS_INFO( "Delta: %f", par_.delta);

  // Yaw-related parameters
  ROS_INFO( "Joint Position/Yaw Optimization: mandatory");
  ROS_INFO( "Alpha Filter Dyaw: %f", par_.alpha_filter_dyaw);
  ROS_INFO( "W Max: %f", par_.w_max);
  ROS_INFO( "Yaw Acc Max: %f", par_.yaw_acc_max);
  ROS_INFO( "Yaw Goal Tolerance: %f",
              par_.yaw_goal_tolerance);
  ROS_INFO( "Yaw Spinning Threshold: %d", par_.yaw_spinning_threshold);
  ROS_INFO( "Yaw Spinning Dyaw: %f", par_.yaw_spinning_dyaw);

  // Simulation env parameters
  ROS_INFO( "Force Goal Z: %d", par_.force_goal_z);
  ROS_INFO( "Default Goal Z: %f", par_.default_goal_z);
  ROS_INFO(
              "Global-once budgets: refinement=%.1fs hard=%.1fs, "
              "minimum interior speed=%.2fm/s",
              par_.global_once_refinement_timeout_seconds,
              par_.global_once_hard_timeout_seconds,
              par_.global_once_minimum_interior_speed);

  ROS_INFO(
      "frontend: shape=%s timeout=%.1fs yaw_cost=%.3fm/rad "
      "resolution=(%.2fm, %.1fdeg) yaw_step<=%.1fdeg "
      "ceiling=%.2fm collision_sample=(%.2fm, %.1fdeg)",
      shape_aware_astar_shape_.c_str(),
      frontend_config_.timeout_seconds,
      frontend_config_.collision.yaw_cost_weight,
      frontend_config_.collision.spatial_resolution,
      frontend_config_.collision.yaw_resolution * 180.0 / M_PI,
      frontend_config_.collision.max_yaw_delta_per_step * 180.0 / M_PI,
      frontend_config_.collision.virtual_ceiling_height,
      frontend_config_.collision.collision_sample_max_distance,
      frontend_config_.collision.collision_sample_max_yaw_step * 180.0 / M_PI);

  // Debug flag
  ROS_INFO( "Debug Verbose: %d", par_.debug_verbose);
}

// ----------------------------------------------------------------------------

/**
 * @brief Callback function to clean up old trajs in DYNUS
 */
void MIGHTY_NODE::stateCallback(const dynus_interfaces::State::ConstPtr msg) {
  if (par_.use_state_update) {
    state current_state;
    current_state.setPos(msg->pos.x, msg->pos.y, msg->pos.z);
    current_state.setVel(msg->vel.x, msg->vel.y, msg->vel.z);
    current_state.setAccel(0.0, 0.0, 0.0);
    double roll, pitch, yaw;
    mighty_utils::quaternion2Euler(msg->quat, roll, pitch, yaw);
    current_state.setYaw(yaw);
    current_state.t = ros::Time::now().toSec();
    mighty_ptr_->updateState(current_state);

    // publish the state
    publishCurrentState(current_state);

    // Publish the velocity in text
    if (par_.visual_level >= 1) publishVelocityInText(current_state.pos, current_state.vel.norm());
  }

  if (!state_initialized_) {
    // If we don't use state update, we need to initialize the state
    if (!par_.use_state_update) {
      state current_state;
      current_state.setPos(msg->pos.x, msg->pos.y, msg->pos.z);
      current_state.setVel(msg->vel.x, msg->vel.y, msg->vel.z);
      current_state.setAccel(0.0, 0.0, 0.0);
      double roll, pitch, yaw;
      mighty_utils::quaternion2Euler(msg->quat, roll, pitch, yaw);
      current_state.setYaw(yaw);
      current_state.t = ros::Time::now().toSec();
      mighty_ptr_->updateState(current_state);
    }
    ROS_INFO( "State initialized");
    state_initialized_ = true;
    timer_goal_.start();
  }

  if (shape_aware_astar_available_) {
    publishShapeAwareAstarCurrentBody(*msg);
  }

  // Throttle actual_traj viz to ~20 Hz to match the replan viz throttle.
  // stateCallback runs at 100 Hz (fake_sim publishes state every 10 ms) and
  // the persistent LINE_STRIP marker is large enough that RViz drops
  // messages with "some messages were lost" warnings without throttling.
  if (par_.visual_level >= 1) {
    const double t_now_actual = ros::Time::now().toSec();
    if (t_now_actual - last_actual_traj_publish_t_ >= 0.05) {
      publishActualTraj();
      last_actual_traj_publish_t_ = t_now_actual;
    }
  }

}


/**
 * @brief Public callback for the terminal goal topic.
 */
void MIGHTY_NODE::terminalGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  terminalGoalCallbackImpl(*msg);
}

/**
 * @brief Validate and execute a global known-map goal.
 */
void MIGHTY_NODE::terminalGoalCallbackImpl(const geometry_msgs::PoseStamped& msg) {
  if (isGlobalOnceMode()) {
    const auto goal_start = mighty::beginGlobalOnceGoal(global_once_phase_);
    if (goal_start != mighty::GlobalOnceGoalStart::STARTED) {
      mighty_ptr_->cancelPendingGlobalPlan();
      if (goal_start == mighty::GlobalOnceGoalStart::SUPERSEDED_EXECUTION) {
        ROS_INFO(
                    "[GLOBAL_ONCE] preempting the executing goal");
      } else {
        ROS_INFO(
                    "[GLOBAL_ONCE] replacing the pending goal");
      }
    }
  }

  // Record the time when goal is received (for command-to-execution timing)
  goal_received_time_ = ros::Time::now();
  waiting_for_first_traj_ = true;

  // Debug log: every received term_goal
  if (auto& s = debug_log_stream(); s.is_open()) {
    s << debug_log_t(ros::Time::now().toSec()) << " TERM_GOAL "
      << msg.pose.position.x << " " << msg.pose.position.y << " " << msg.pose.position.z
      << " frame=" << msg.header.frame_id
      << "\n";
    s.flush();
  }

  // Set the terminal goal
  state G_term;
  double goal_z;

  // If force_goal_z is true, set the goal_z to default_goal_z
  if (par_.force_goal_z)
    goal_z = par_.default_goal_z;
  else
    goal_z = msg.pose.position.z;

  // Set the terminal goal
  G_term.setPos(msg.pose.position.x, msg.pose.position.y, goal_z);
  if (isGlobalOnceMode()) {
    const auto& orientation = msg.pose.orientation;
    const std::array<double, 4> quaternion{
        orientation.x, orientation.y, orientation.z, orientation.w};
    if (!std::all_of(quaternion.begin(), quaternion.end(),
                     [](double value) { return std::isfinite(value); })) {
      waiting_for_first_traj_ = false;
      global_once_phase_ = mighty::GlobalOncePhase::IDLE;
      ROS_ERROR(
                   "[GLOBAL_ONCE] REJECTED: stage=invalid_goal_orientation");
      return;
    }
    const double norm = std::sqrt(
        orientation.x * orientation.x + orientation.y * orientation.y +
        orientation.z * orientation.z + orientation.w * orientation.w);
    if (norm <= 1e-12) {
      G_term.use_tracking_yaw = false;
    } else {
      geometry_msgs::Quaternion normalized = orientation;
      normalized.x /= norm;
      normalized.y /= norm;
      normalized.z /= norm;
      normalized.w /= norm;
      double roll, pitch, yaw;
      mighty_utils::quaternion2Euler(normalized, roll, pitch, yaw);
      G_term.yaw = yaw;
      G_term.use_tracking_yaw = true;
    }
  }

  if (isGlobalOnceMode()) {
    ROS_INFO(
                "[GLOBAL_ONCE] RECEIVED: goal=(%.3f, %.3f, %.3f) "
                "yaw=%s%.6f",
                G_term.pos.x(), G_term.pos.y(), G_term.pos.z(),
                G_term.use_tracking_yaw ? "" : "unspecified/",
                G_term.yaw);
    // Invalidate any pre-commit visualization from an older request. Shape A*
    // now runs synchronously inside the core planning transaction.
    cancelPendingShapeAwareAstar();
    const std::uint64_t shape_generation =
        shape_aware_astar_generation_.fetch_add(1) + 1;
    publishShapeAwareAstarStatus("running", shape_generation);
    setComputationTimesToZero();
    const double start_time = ros::Time::now().toSec();
    const auto result = mighty_ptr_->planGlobalOnce(G_term, start_time);
    const double total_ms = (ros::Time::now().toSec() - start_time) * 1000.0;
    if (shape_generation != shape_aware_astar_generation_.load()) {
      ROS_INFO(
                  "[GLOBAL_ONCE] discarded superseded planning result");
      return;
    }
    retrieveData();
    publishComputationTimes(total_ms);
    publishMiddleStageTime(result);
    publishGlobalPlanStatus(result);
    if (par_.visual_level >= 1) publishTraj();

    if (result.shape_path.success()) {
      publishShapeAwareAstarResult(result.shape_path, shape_generation);
      publishShapeAwareAstarStatus("success", shape_generation);
    } else {
      clearShapeAwareAstarPlanVisualization(shape_generation);
      std::ostringstream shape_status;
      shape_status << "failure:" << frontend::toString(result.shape_path.status);
      publishShapeAwareAstarStatus(shape_status.str(), shape_generation);
    }

    if (!result.success()) {
      const char* stage = "unknown";
      switch (result.stage) {
        case GlobalPlanStage::NOT_READY: stage = "not_ready"; break;
        case GlobalPlanStage::SHAPE_ASTAR_FAILED: stage = "shape_astar"; break;
        case GlobalPlanStage::OPTIMIZATION_FAILED: stage = "optimization"; break;
        default: break;
      }
      waiting_for_first_traj_ = false;
      global_once_phase_ = mighty::GlobalOncePhase::IDLE;
      if (result.stage == GlobalPlanStage::SHAPE_ASTAR_FAILED) {
        const auto& st = result.shape_path.statistics;
        ROS_ERROR(
                     "[GLOBAL_ONCE] REJECTED: stage=%s goal=(%.3f, %.3f, %.3f); "
                     "%s (expansions=%zu edge_checks=%zu %.2fs)",
                     stage, G_term.pos.x(), G_term.pos.y(), G_term.pos.z(),
                     result.message.empty() ? "goal dropped"
                                            : result.message.c_str(),
                     st.expansions, st.edge_checks, st.elapsed_seconds);
      } else {
        ROS_ERROR(
                     "[GLOBAL_ONCE] REJECTED: stage=%s goal=(%.3f, %.3f, %.3f); "
                     "%s",
                     stage, G_term.pos.x(), G_term.pos.y(), G_term.pos.z(),
                     result.message.empty() ? "goal dropped"
                                            : result.message.c_str());
      }
      return;
    }

    publishState(G_term, pub_point_G_term_);
    if (result.stage == GlobalPlanStage::ALREADY_REACHED) {
      waiting_for_first_traj_ = false;
      global_once_phase_ = mighty::GlobalOncePhase::IDLE;
      pub_goal_reached_.publish(std_msgs::Empty());
      ROS_INFO(
                  "[GLOBAL_ONCE] COMPLETE: goal is already within goal_radius");
      return;
    }

    global_once_phase_ = mighty::GlobalOncePhase::EXECUTING;
    replanning_computation_time_ = total_ms / 1000.0;
    publishTrajectory();
    publishMpcPath();

    std_msgs::Float64 time_msg;
    time_msg.data = (ros::Time::now() - goal_received_time_).toSec() * 1000.0;
    pub_command_to_exec_time_.publish(time_msg);
    waiting_for_first_traj_ = false;

    if (par_.visual_level >= 1) {
      publishPointG();
      publishPointE();
      publishPointA();
      publishCps();
    }
    const char* outcome = globalPlanOutcomeName(result.outcome);
    const char* route = trajectoryRouteName(result.route);
    if (result.outcome == GlobalPlanOutcome::SAFE_DEGRADED) {
      ROS_WARN(
                  "[GLOBAL_ONCE] COMMITTED DEGRADED: outcome=%s route=%s "
                  "setpoints=%zu total=%.2fms min_interior_speed=%.4fm/s "
                  "max_low_speed_duration=%.2fs",
                  outcome, route, goal_setpoints_.size(), total_ms,
                  result.quality.minimum_interior_speed,
                  result.quality.maximum_subthreshold_duration);
    } else {
      ROS_INFO(
                  "[GLOBAL_ONCE] COMMITTED: outcome=%s route=%s "
                  "setpoints=%zu total=%.2fms min_interior_speed=%.4fm/s",
                  outcome, route, goal_setpoints_.size(), total_ms,
                  result.quality.minimum_interior_speed);
    }
    return;
  }
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish velocity in text
 * @param position Position to publish the text
 * @param velocity Velocity to publish
 */
void MIGHTY_NODE::publishVelocityInText(const Eigen::Vector3d& position, double velocity) {
  // Set velocity's precision to 2 decimal points
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << velocity;

  // Make a string
  std::string text = oss.str() + "m/s";

  visualization_msgs::Marker marker;
  marker.header.frame_id = par_.map_frame_id;
  marker.header.stamp = ros::Time::now();
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.ns = "velocity";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.scale.z = 1.0;
  marker.color.a = 1.0;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.text = text;
  marker.pose.position.x = position.x();
  marker.pose.position.y = position.y();
  marker.pose.position.z = position.z() + 5.0;
  marker.pose.orientation.w = 1.0;
  pub_vel_text_.publish(marker);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishFsmStatus() {
  if (!mighty_ptr_) return;
  std_msgs::UInt8 msg;
  msg.data = static_cast<uint8_t>(mighty_ptr_->getDroneStatus());
  pub_fsm_status_.publish(msg);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishComputationTimes(double total_ms) {
  dynus_interfaces::ComputationTimes msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = par_.map_frame_id;
  msg.total_planning_ms = total_ms;
  msg.frontend_planning_ms = global_planning_time_;
  msg.trajectory_optimization_ms = trajectory_optimization_time_;
  pub_computation_times_.publish(msg);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishMiddleStageTime(const GlobalPlanResult& result) {
  std_msgs::Float64 message;
  message.data = result.has_trajectory()
                     ? result.middle_time_ms
                     : std::numeric_limits<double>::quiet_NaN();
  pub_middle_stage_time_.publish(message);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishGlobalPlanStatus(const GlobalPlanResult& result) {
  dynus_interfaces::GlobalPlanStatus msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = par_.map_frame_id;
  msg.outcome = globalPlanOutcomeMessage(result.outcome);
  msg.stage = globalPlanStageMessage(result.stage);
  msg.route = trajectoryRouteMessage(result.route);
  msg.reason = result.message;
  if (result.outcome == GlobalPlanOutcome::SAFE_DEGRADED) {
    if (!msg.reason.empty()) msg.reason += "; ";
    msg.reason += "interior speed dropped below 0.05 m/s";
  }
  msg.minimum_interior_speed = result.quality.minimum_interior_speed;
  msg.maximum_subthreshold_duration =
      result.quality.maximum_subthreshold_duration;
  pub_global_plan_status_.publish(msg);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishGlobalPlanStatus(bool success) {
  dynus_interfaces::GlobalPlanStatus msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = par_.map_frame_id;
  msg.outcome = success
                    ? dynus_interfaces::GlobalPlanStatus::NOMINAL
                    : dynus_interfaces::GlobalPlanStatus::FAILED;
  msg.stage = success
                  ? dynus_interfaces::GlobalPlanStatus::STAGE_SUCCESS
                  : dynus_interfaces::GlobalPlanStatus::STAGE_OPTIMIZATION_FAILED;
  msg.route = dynus_interfaces::GlobalPlanStatus::ROUTE_NONE;
  if (!success) msg.reason = "rolling replanning failed";
  pub_global_plan_status_.publish(msg);
}

// ----------------------------------------------------------------------------

/**
 * @brief Callback function to check if the goal is reached
 */
void MIGHTY_NODE::goalReachedCheckCallback() {
  if (isGlobalOnceMode() &&
      global_once_phase_.load() != mighty::GlobalOncePhase::EXECUTING) {
    return;
  }
  if (!mighty_ptr_->goalReachedCheck(ros::Time::now().toSec())) return;

  if (isGlobalOnceMode()) {
    if (!mighty::completeGlobalOnceGoal(global_once_phase_)) return;
    ROS_INFO(
                "[GLOBAL_ONCE] COMPLETE: trajectory elapsed; position and yaw satisfied");
  }

  pub_goal_reached_.publish(std_msgs::Empty());

}

// ----------------------------------------------------------------------------

/**
 * @brief Callback function to get the initial pose from tf (for hardware use case)
 */
void MIGHTY_NODE::publishCps() {
  // Retrieve control points
  mighty_ptr_->retrieveCPs(cps_);

  // Create a marker array
  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.resize(cps_.size());

  // Loop through the control points (std::vector<Eigen::Matrix<double, 3, 4>>)
  for (int seg = 0; seg < cps_.size(); seg++) {
    // Create a marker
    visualization_msgs::Marker marker;
    marker.header.frame_id = par_.map_frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "cp";
    marker.id = seg;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 0.5;
    marker.color.a = 1.0;

    // Set different colors for different segments
    if (seg % 3 == 0) {
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
    } else if (seg % 3 == 1) {
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
    } else {
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
    }

    auto cp = cps_[seg];

    // Loop through the control points for each segment
    for (int i = 0; i < cp.cols(); i++) {
      geometry_msgs::Point point;
      point.x = cp(0, i);
      point.y = cp(1, i);
      point.z = cp(2, i);
      marker.points.push_back(point);
    }

    // Add the marker to the marker array
    marker_array.markers[seg] = marker;
  }

  // Publish
  pub_cp_.publish(marker_array);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish static push points
 */
void MIGHTY_NODE::publishStaticPushPoints() {
  // Create a marker array
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  marker.header.frame_id = par_.map_frame_id;
  marker.header.stamp = ros::Time::now();
  marker.ns = "static_push_points";
  marker.id = static_push_points_id_;
  marker.type = visualization_msgs::Marker::SPHERE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 3.0;
  marker.scale.y = 3.0;
  marker.scale.z = 3.0;
  marker.color.a = 1.0;
  marker.color.r = 0.0;
  marker.color.g = 0.925;
  marker.color.b = 1.0;

  // Loop through the static push points
  for (int idx = 0; idx < static_push_points_.size(); idx++) {
    geometry_msgs::Point point;
    point.x = static_push_points_[idx](0);
    point.y = static_push_points_[idx](1);
    point.z = static_push_points_[idx](2);
    marker.points.push_back(point);
  }

  // Add the single marker to the marker array
  marker_array.markers.push_back(marker);

  // Publish
  pub_static_push_points_.publish(marker_array);
}

// ----------------------------------------------------------------------------

/**
 * @brief Set computation times to zero
 */
void MIGHTY_NODE::setComputationTimesToZero() {
  final_g_ = 0.0;
  global_planning_time_ = 0.0;
  initial_guess_computation_time_ = 0.0;
  trajectory_optimization_time_ = 0.0;
}

// ----------------------------------------------------------------------------

/**
 * @brief Retrive computation times from mighty_ptr_
 */
void MIGHTY_NODE::retrieveData() {
  mighty_ptr_->retrieveData(final_g_, global_planning_time_,
                            initial_guess_computation_time_,
                            trajectory_optimization_time_);
}

// ----------------------------------------------------------------------------

/**
 * @brief Print the computation times
 */
void MIGHTY_NODE::printComputationTime(bool result) {
  // Print the computation times
  ROS_INFO( "Planner: Shape-aware Weighted A*");
  ROS_INFO( "Result: %d", result);
  ROS_INFO( "Cost (final node's g): %f", final_g_);
  ROS_INFO( "Total replanning time [ms]: %f",
              replanning_computation_time_ * 1000.0);
  ROS_INFO( "Global Planning Time [ms]: %f", global_planning_time_);
  ROS_INFO( "Initial Guess Time [ms]: %f", initial_guess_computation_time_);
  ROS_INFO( "Trajectory Optimization Time [ms]: %f",
              trajectory_optimization_time_);
  ROS_INFO( "------------------------");
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish the Point G (sub goal)
 */
void MIGHTY_NODE::publishPointG() const {
  state G;
  mighty_ptr_->getG(G);
  publishState(G, pub_point_G_);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish the Point E (sub goal)
 */
void MIGHTY_NODE::publishPointE() const {
  // get projected goal (E)
  state E;
  mighty_ptr_->getE(E);

  // Publish the goal for visualization
  publishState(E, pub_point_E_);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish the Point A (trajectory start point)
 */
void MIGHTY_NODE::publishPointA() const {
  // get projected goal (A)
  state A;
  mighty_ptr_->getA(A);

  // Publish the goal for visualization
  publishState(A, pub_point_A_);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish the current state
 */
void MIGHTY_NODE::publishCurrentState(const state& state) const {
  // Publish the goal for visualization
  publishState(state, pub_current_state_);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish state
 */
void MIGHTY_NODE::publishState(
    const state& data,
    const ros::Publisher& publisher) const {
  geometry_msgs::PointStamped p;
  p.header.frame_id = par_.map_frame_id;
  p.header.stamp = ros::Time::now();
  p.point = eigen2point(data.pos);
  publisher.publish(p);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish its own trajectory for deconfliction
 */
void MIGHTY_NODE::publishActualTraj() {
  if (!pub_actual_traj_) return;

  state current_state;
  mighty_ptr_->getState(current_state);
  const Eigen::Vector3d current_pos = current_state.pos;

  // Skip until the state is actually populated.
  if (current_pos.norm() < 1e-2) return;

  const auto now = ros::Time::now();
  const double tnow = now.toSec();

  // Initialize on first valid sample.
  if (!actual_traj_initialized_) {
    actual_traj_prev_pos_  = current_pos;
    actual_traj_prev_time_ = tnow;

    actual_traj_hist_.clear();
    actual_traj_hist_.push_back(current_state);
    actual_traj_initialized_ = true;
    return;  // wait for second sample to draw a line
  }

  actual_traj_prev_pos_  = current_pos;
  actual_traj_prev_time_ = tnow;

  // Append to history only if it moved enough — prevents dense duplicates
  // when the robot is stationary. Still update the latest sample's velocity
  // so colors stay accurate.
  const double eps = 1e-3;
  if (!actual_traj_hist_.empty()) {
    const Eigen::Vector3d last_pos = actual_traj_hist_.back().pos;
    if ((current_pos - last_pos).norm() < eps) {
      actual_traj_hist_.back().vel = current_state.vel;
    } else {
      actual_traj_hist_.push_back(current_state);
    }
  } else {
    actual_traj_hist_.push_back(current_state);
  }

  // Bound history to keep RViz responsive.
  if (actual_traj_hist_.size() > actual_traj_max_hist_) {
    const size_t overflow = actual_traj_hist_.size() - actual_traj_max_hist_;
    actual_traj_hist_.erase(actual_traj_hist_.begin(),
                            actual_traj_hist_.begin() + overflow);
  }

  // Build a single persistent colored LINE_STRIP. par_.v_max is per-axis;
  // matching sando we use it directly as the jet-colormap max.
  const double vmax_for_color = par_.v_max;
  visualization_msgs::MarkerArray ma = stateVector2ColoredLineStripMarkerArray(
      actual_traj_hist_,
      /*id=*/1,
      /*ns=*/"actual_traj_" + id_str_,
      /*max_value=*/vmax_for_color,
      /*stamp=*/now,
      /*line_width=*/actual_traj_line_width_,
      /*max_points_vis=*/actual_traj_max_points_vis_);

  pub_actual_traj_.publish(ma);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish goal (setpoint)
 */
void MIGHTY_NODE::publishGoal() {
  // Initialize the goal
  state next_goal;

  // Get the next goal
  if (mighty_ptr_->getNextGoal(ros::Time::now().toSec(), next_goal) &&
      par_.use_state_update) {
    // Publish the goal (actual setpoint)
    dynus_interfaces::Goal quadGoal;
    quadGoal.header.stamp = ros::Time::now();
    quadGoal.header.frame_id = par_.map_frame_id;
    quadGoal.p = eigen2rosvector(next_goal.pos);
    quadGoal.v = eigen2rosvector(next_goal.vel);
    quadGoal.a = eigen2rosvector(next_goal.accel);
    quadGoal.j = eigen2rosvector(next_goal.jerk);
    quadGoal.yaw = next_goal.yaw;
    quadGoal.dyaw = next_goal.dyaw;
    pub_goal_.publish(quadGoal);

    // Debug log: every published setpoint, with current state and plan front
    // for context. Helps trace dip-toward-origin bugs.
    if (auto& s = debug_log_stream(); s.is_open()) {
      state cur, gterm, gsub;
      mighty_ptr_->getState(cur);
      mighty_ptr_->getGterm(gterm);
      mighty_ptr_->getG(gsub);
      s << debug_log_t(ros::Time::now().toSec()) << " GOAL "
        << next_goal.pos.x() << " " << next_goal.pos.y() << " " << next_goal.pos.z()
        << " state=" << cur.pos.x() << "," << cur.pos.y() << "," << cur.pos.z()
        << " G=" << gsub.pos.x() << "," << gsub.pos.y() << "," << gsub.pos.z()
        << " Gterm=" << gterm.pos.x() << "," << gterm.pos.y() << "," << gterm.pos.z()
        << "\n";
      s.flush();
    }

    // Publish the goal (setpoint) for visualization
    if (par_.visual_level >= 1) publishState(next_goal, pub_setpoint_);
  }

}

// ----------------------------------------------------------------------------

/**
 * @brief Publish the full trajectory for robust tracking with replan support
 */
void MIGHTY_NODE::publishTrajectory() {
  // Retrieve the goal setpoints (full trajectory)
  mighty_ptr_->retrieveGoalSetpoints(goal_setpoints_);

  if (goal_setpoints_.empty()) {
    ROS_WARN_THROTTLE(1.0,
                         "Cannot publish trajectory: goal_setpoints_ is empty");
    return;
  }

  // Increment trajectory_id for new trajectory (replan detected by change in ID)
  trajectory_id_++;

  // Downsample trajectory: publish ~50 points instead of all points
  const size_t target_points = par_.trajectory_downsample_points;
  size_t total_points = goal_setpoints_.size();
  size_t step = std::max(size_t(1), total_points / target_points);

  // Create trajectory message
  dynus_interfaces::Trajectory traj_msg;
  traj_msg.header.stamp = ros::Time::now();
  traj_msg.header.frame_id = par_.map_frame_id;
  traj_msg.trajectory_id = trajectory_id_;
  traj_msg.dt = par_.dc * step;  // Adjusted time step for downsampled trajectory

  // Downsample: take every 'step' points
  traj_msg.goals.reserve(target_points);
  for (size_t i = 0; i < goal_setpoints_.size(); i += step) {
    const auto& state_point = goal_setpoints_[i];
    dynus_interfaces::Goal goal;
    goal.p = eigen2rosvector(state_point.pos);
    goal.v = eigen2rosvector(state_point.vel);
    goal.a = eigen2rosvector(state_point.accel);
    goal.j = eigen2rosvector(state_point.jerk);
    goal.yaw = state_point.yaw;
    goal.dyaw = state_point.dyaw;
    traj_msg.goals.push_back(goal);
  }

  // Always include the last point
  if (!goal_setpoints_.empty() && (goal_setpoints_.size() - 1) % step != 0) {
    const auto& last_point = goal_setpoints_.back();
    dynus_interfaces::Goal goal;
    goal.p = eigen2rosvector(last_point.pos);
    goal.v = eigen2rosvector(last_point.vel);
    goal.a = eigen2rosvector(last_point.accel);
    goal.j = eigen2rosvector(last_point.jerk);
    goal.yaw = last_point.yaw;
    goal.dyaw = last_point.dyaw;
    traj_msg.goals.push_back(goal);
  }

  // Publish trajectory
  pub_trajectory_.publish(traj_msg);
}

// ----------------------------------------------------------------------------

void MIGHTY_NODE::publishMpcPath() {
  // Get optimized trajectory setpoints (smooth quintic Hermite spline)
  mighty_ptr_->retrieveGoalSetpoints(goal_setpoints_);

  if (goal_setpoints_.size() < 2) return;

  dynus_interfaces::SpeedyPath path_msg;
  path_msg.header.stamp = ros::Time::now();
  path_msg.header.frame_id = par_.map_frame_id;

  const double spacing = par_.mpc_path_spacing;

  // Collect speeds alongside poses
  std::vector<double> speeds;
  double last_emitted_time = -std::numeric_limits<double>::infinity();

  // Always include first point
  auto addPose = [&](const state& sp) {
    geometry_msgs::PoseStamped pose;
    pose.header = path_msg.header;
    pose.pose.position.x = sp.pos.x();
    pose.pose.position.y = sp.pos.y();
    pose.pose.position.z = sp.pos.z();
    if (isGlobalOnceMode()) {
      pose.pose.orientation.z = std::sin(sp.yaw / 2.0);
      pose.pose.orientation.w = std::cos(sp.yaw / 2.0);
    } else {
      pose.pose.orientation.w = 1.0;
    }
    path_msg.poses.push_back(pose);
    speeds.push_back(sp.vel.head<2>().norm());
    last_emitted_time = sp.t;
  };

  addPose(goal_setpoints_.front());
  Eigen::Vector3d last_emitted = goal_setpoints_.front().pos;

  // Walk through setpoints, emit when distance threshold exceeded
  for (size_t i = 1; i < goal_setpoints_.size(); ++i) {
    double dist = (goal_setpoints_[i].pos - last_emitted).head<2>().norm();
    if (dist >= spacing) {
      addPose(goal_setpoints_[i]);
      last_emitted = goal_setpoints_[i].pos;
    }
  }

  // Always include last point
  const auto& last = goal_setpoints_.back();
  if ((last.pos - last_emitted).head<2>().norm() > 1e-6 ||
      (isGlobalOnceMode() &&
       std::abs(last.t - last_emitted_time) > 1e-9)) {
    addPose(last);
  }

  // Need at least 2 points for MPC
  if (path_msg.poses.size() < 2) return;

  if (!isGlobalOnceMode()) {
    // Legacy behavior: infer yaw from the spatial path direction.
    for (size_t i = 0; i < path_msg.poses.size(); ++i) {
      double yaw = 0.0;
      if (i + 1 < path_msg.poses.size()) {
        double dx = path_msg.poses[i + 1].pose.position.x -
                    path_msg.poses[i].pose.position.x;
        double dy = path_msg.poses[i + 1].pose.position.y -
                    path_msg.poses[i].pose.position.y;
        yaw = std::atan2(dy, dx);
      } else if (i > 0) {
        double dx = path_msg.poses[i].pose.position.x -
                    path_msg.poses[i - 1].pose.position.x;
        double dy = path_msg.poses[i].pose.position.y -
                    path_msg.poses[i - 1].pose.position.y;
        yaw = std::atan2(dy, dx);
      }
      path_msg.poses[i].pose.orientation.x = 0.0;
      path_msg.poses[i].pose.orientation.y = 0.0;
      path_msg.poses[i].pose.orientation.z = std::sin(yaw / 2.0);
      path_msg.poses[i].pose.orientation.w = std::cos(yaw / 2.0);
    }
  }

  // Assign per-waypoint speeds
  path_msg.speeds = speeds;

  pub_mpc_path_.publish(path_msg);
}

// ----------------------------------------------------------------------------

/**
 * @brief Publish the trajectory
 */
void MIGHTY_NODE::publishTraj() {
  auto now = ros::Time::now();

  // Build ONE MarkerArray that contains DELETEALL followed by the new markers,
  // for each topic. Publishing DELETEALL and the new markers as two separate
  // messages was racy: if the second message was dropped or arrived out of
  // order, RViz would show an empty topic. A single MarkerArray is processed
  // atomically — RViz applies DELETEALL first, then the ADDs that follow.
  auto deleteAll = [&]() {
    visualization_msgs::Marker m;
    m.header.frame_id = par_.map_frame_id;
    m.header.stamp    = ros::Time(0);  // latest TF
    m.action          = visualization_msgs::Marker::DELETEALL;
    return m;
  };

  // 1) Certified middle-stage warm start, before backend refinement.
  std::vector<state> middle_goal_setpoints;
  mighty_ptr_->retrieveMiddleGoalSetpoints(middle_goal_setpoints);
  {
    visualization_msgs::MarkerArray middle_ma;
    middle_ma.markers.push_back(deleteAll());
    auto added = stateVector2ColoredMarkerArray(
        middle_goal_setpoints, /*type=*/1, par_.v_max, now, par_.map_frame_id);
    for (auto& marker : added.markers) {
      marker.ns = "trajectory_middle";
      marker.scale.x = 0.09;
      for (auto& color : marker.colors) {
        color.r = 1.0F;
        color.g = 0.55F;
        color.b = 0.0F;
        color.a = 1.0F;
      }
    }
    middle_ma.markers.insert(middle_ma.markers.end(),
                             added.markers.begin(), added.markers.end());
    pub_traj_middle_.publish(middle_ma);
  }

  // 2) Committed backend trajectory.
  mighty_ptr_->retrieveGoalSetpoints(goal_setpoints_);
  {
    visualization_msgs::MarkerArray committed_ma;
    committed_ma.markers.push_back(deleteAll());
    auto added = stateVector2ColoredMarkerArray(
        goal_setpoints_, /*type=*/1, par_.v_max, now, par_.map_frame_id);
    committed_ma.markers.insert(committed_ma.markers.end(),
                                added.markers.begin(), added.markers.end());
    pub_traj_committed_colored_.publish(committed_ma);
    if (isGlobalOnceMode()) pub_traj_backend_.publish(committed_ma);
  }

}

// ----------------------------------------------------------------------------

/**
 * @brief Create MarkerArray from vec_Vec3f
 */
void MIGHTY_NODE::createMarkerArrayFromVec_Vec3f(
    const vec_Vec3f& occupied_cells, const std_msgs::ColorRGBA& color, int namespace_id,
    double scale, visualization_msgs::MarkerArray* marker_array) {
  visualization_msgs::Marker marker;
  marker.header.frame_id = par_.map_frame_id;
  marker.header.stamp = ros::Time::now();
  marker.ns = "namespace_" + std::to_string(namespace_id);
  marker.pose.orientation.w = 1.0;
  marker.id = 0;
  marker.type =
      visualization_msgs::Marker::CUBE_LIST;  // Each point will be visualized as a cube
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = par_.known_map_resolution;
  marker.scale.y = par_.known_map_resolution;
  marker.scale.z = par_.known_map_resolution;
  marker.color = color;

  for (const auto& cell : occupied_cells) {
    geometry_msgs::Point point;
    point.x = cell(0);
    point.y = cell(1);
    point.z = cell(2);
    marker.points.push_back(point);
  }

  marker_array->markers.push_back(marker);
}

// ----------------------------------------------------------------------------

/**
 * @brief Clear any marker array
 */
void MIGHTY_NODE::clearMarkerArray(
    visualization_msgs::MarkerArray& path_marker,
    ros::Publisher publisher) {
  // If the marker array is empty, return
  if (path_marker.markers.size() == 0) return;

  // Clear the marker array
  int id_begin = path_marker.markers[0].id;

  for (int i = 0; i < path_marker.markers.size(); i++) {
    visualization_msgs::Marker m;
    m.type = visualization_msgs::Marker::ARROW;
    m.action = visualization_msgs::Marker::DELETE;
    m.id = i + id_begin;
    path_marker.markers[i] = m;
  }

  publisher.publish(path_marker);
  path_marker.markers.clear();
}

}  // namespace mighty
