#include <mighty/frontend/types.hpp>
#include <mighty/frontend/collision_scene.hpp>
#include <mighty/mighty.hpp>
#include <mighty/whole_body_safety.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <csignal>
#include <sys/time.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

struct Request {
  std::string name;
  std::string map;
  Eigen::Vector3d start;
  double start_yaw{0.0};
  Eigen::Vector3d goal;
  double goal_yaw{0.0};
  std::string shape{"t"};
  bool goal_yaw_exact{true};
};

parameters makeParameters(const std::string& map, bool risk_local,
                          bool spatial_progress_yaw = true,
                          bool risk_window = true) {
  parameters par{};
  par.sim_env = "fake_sim";
  par.planning_mode = "global_once";
  par.map_mode = "known_pcd";
  par.visual_level = 0;
  par.v_max = 2.0;
  par.a_max = 5.0;
  par.j_max = 10.0;
  par.w_max = 1.0;
  par.yaw_acc_max = 2.0;
  par.drone_bbox = {0.5, 0.5, 0.5};
  par.drone_radius = 0.25;
  par.goal_radius = 0.5;
  par.time_weight = 5.0e2;
  par.stat_weight = 1.0e3;
  par.jerk_weight = 10.0;
  par.yaw_jerk_weight = 10.0;
  par.yaw_reference_weight = 10.0;
  par.dyn_constr_vel_weight = 1.0e3;
  par.dyn_constr_acc_weight = 1.0e3;
  par.dyn_constr_jerk_weight = 1.0e3;
  par.planner_Co_normal = 0.1;
  par.planner_Co_small = 0.05;
  par.planner_Co = map == "hole" ? par.planner_Co_small : par.planner_Co_normal;
  par.risk_local_refinement = risk_local;
  par.spatial_progress_yaw_coupling = spatial_progress_yaw;
  par.risk_window_refinement = risk_window;
  par.max_dist_vertexes = 2.0;
  par.spline_degree = 5;
  par.dc = 0.01;
  par.pos_anchor_weight = 0.0;
  par.f_dec_coeff = 1.0e-3;
  par.cautious_factor = 1.0e-6;
  par.past = 3;
  par.max_linesearch = 64;
  par.max_iterations = 1000;
  par.g_epsilon = 1.0e-5;
  par.delta = 1.0e-3;
  par.global_once_refinement_timeout_seconds = 2.9;
  par.global_once_hard_timeout_seconds = 5.0;
  par.global_once_minimum_interior_speed = 0.2;
  par.use_state_update = true;
  par.force_goal_z = false;
  par.default_goal_z = 1.0;
  par.debug_verbose = false;
  par.trajectory_downsample_points = 500;
  par.mpc_path_spacing = 0.05;
  if (map == "narrow") { par.x_min=-10; par.x_max=10; par.y_min=-2; par.y_max=2; par.z_min=-.3; par.z_max=3.3; }
  else if (map == "forest") { par.x_min=-15; par.x_max=15; par.y_min=-15; par.y_max=15; par.z_min=-.1; par.z_max=4; }
  else if (map == "hole") { par.x_min=-7.6; par.x_max=7.6; par.y_min=-7.6; par.y_max=7.6; par.z_min=0; par.z_max=3; }
  else { par.x_min=-7.5; par.x_max=7.5; par.y_min=-7.5; par.y_max=7.5; par.z_min=0; par.z_max=3; }
  return par;
}

mighty::frontend::FrontendConfig makeFrontendConfig(const std::string& map = {}) {
  mighty::frontend::FrontendConfig config;
  config.timeout_seconds = 1.0;
  config.max_expansions = 250000;
  config.heuristic_weight = 1.5;
  config.max_yaw_candidates = 5;
  config.coarse_spatial_resolution = 0.4;
  config.collision.body_clearance = map == "hole" ? 0.05 : 0.1;
  config.collision.collision_sample_max_distance = 0.2;
  config.collision.collision_sample_max_yaw_step = M_PI / 18.0;
  config.collision.virtual_ceiling_height = 3.5;
  config.collision.numeric_tolerance = 1.0e-8;
  config.collision.yaw_cost_weight = 0.2;
  config.collision.spatial_resolution = 0.2;
  config.collision.max_yaw_delta_per_step = M_PI / 2.0;
  config.collision.yaw_resolution = M_PI / 18.0;
  config.collision.smoothing_enabled = true;
  config.collision.smoothing_iterations = 4;
  config.collision.smoothing_alpha = 0.35;
  return config;
}

GlobalPlanResult run(const std::filesystem::path& root, const Request& request,
                     bool risk_local) {
  const auto map_path = root / "maps" / "polymap" /
      (request.map + ".polymap.json");
  const auto body_path = root / "shape" / (request.map == "hole" ? "small" : "normal") / "ply" /
      (request.shape + ".polymap.json");
  MIGHTY planner(makeParameters(request.map, risk_local),
                 makeFrontendConfig(request.map),
                 map_path.string(), body_path.string());
  state start{};
  start.pos = request.start;
  start.yaw = request.start_yaw;
  planner.updateState(start);
  state goal{};
  goal.pos = request.goal;
  goal.yaw = request.goal_yaw;
  goal.use_tracking_yaw = request.goal_yaw_exact;
  return planner.planGlobalOnce(goal, 0.0);
}

void printResult(const Request& request, bool risk_local,
                 const GlobalPlanResult& result) {
  std::cout << std::fixed << std::setprecision(3)
            << request.name << ",policy="
            << (risk_local ? "risk_local" : "whole_path")
            << ",success=" << result.success()
            << ",backend_ms=" << result.backend_time_ms
            << ",frontend_expansions="
            << result.shape_path.statistics.expansions
            << ",frontend_collision_queries="
            << result.shape_path.statistics.collision_queries
            << ",frontend_phase_expansions=";
  for (const auto& phase : result.shape_path.statistics.search_phases) {
    std::cout << (phase.ran ? phase.expansions : 0) << ':';
  }
  std::cout
            << ",risk_segments=" << result.risk_segment_count
            << ",risk_edges=" << result.risk_segment_edges
            << ",path_length=" << result.quality.path_length
            << ",duration_s=" << result.quality.duration_seconds
            << ",route=" << static_cast<int>(result.route)
            << ",message=" << result.message << '\n';
}

struct Measurements {
  std::vector<GlobalPlanResult> runs;
  bool all_success{true};
  double p95_backend_ms{0.0};
  double maximum_path_length{0.0};
  double maximum_duration_s{0.0};
};

Measurements measure(const std::filesystem::path& root, const Request& request,
                     bool risk_local) {
  // Cold planners match the production one-request lifecycle.  The first run
  // is deliberately excluded so scene and allocator warm-up do not enter P95.
  static_cast<void>(run(root, request, risk_local));
  Measurements measurements;
  measurements.runs.reserve(10);
  for (int iteration = 0; iteration < 10; ++iteration) {
    auto result = run(root, request, risk_local);
    measurements.all_success = measurements.all_success && result.success();
    if (result.success()) {
      measurements.maximum_path_length = std::max(
          measurements.maximum_path_length, result.quality.path_length);
      measurements.maximum_duration_s = std::max(
          measurements.maximum_duration_s, result.quality.duration_seconds);
    }
    measurements.runs.push_back(std::move(result));
  }
  for (const auto& result : measurements.runs) {
    measurements.p95_backend_ms = std::max(
        measurements.p95_backend_ms, result.backend_time_ms);
  }
  return measurements;
}

void printMeasurements(const Request& request, bool risk_local,
                       const Measurements& measurements) {
  std::cout << std::fixed << std::setprecision(3)
            << request.name << ",policy="
            << (risk_local ? "risk_local" : "whole_path")
            << ",runs=" << measurements.runs.size()
            << ",all_success=" << measurements.all_success
            << ",p95_backend_ms=" << measurements.p95_backend_ms
            << ",max_path_length=" << measurements.maximum_path_length
            << ",max_duration_s=" << measurements.maximum_duration_s
            << '\n';
}

int benchmarkBlock(const std::filesystem::path& root) {
  const std::array<Request, 3> requests{{
      {"block_zero_risk", "block", {3.5, -2.5, 1.0}, 0.0,
       {4.5, -2.5, 1.0}, 0.0},
      {"block_one_risk", "block", {-3.5, -4.5, 1.0}, 0.0,
       {0.5, -0.5, 1.0}, 0.0},
      {"block_multi_risk", "block", {-3.5, -4.5, 1.0}, 0.0,
       {-1.5, 3.5, 1.0}, 0.0},
  }};
  std::array<Measurements, requests.size()> whole_path;
  std::array<Measurements, requests.size()> risk_local;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    whole_path[index] = measure(root, requests[index], false);
    risk_local[index] = measure(root, requests[index], true);
    printMeasurements(requests[index], false, whole_path[index]);
    printMeasurements(requests[index], true, risk_local[index]);
  }

  bool all_success = true;
  double worst_quality_ratio = 1.0;
  double whole_refined_p95_ms = 0.0;
  double local_refined_p95_ms = 0.0;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    all_success = all_success && whole_path[index].all_success &&
        risk_local[index].all_success;
    if (whole_path[index].maximum_path_length > 0.0) {
      worst_quality_ratio = std::max(worst_quality_ratio,
          risk_local[index].maximum_path_length /
              whole_path[index].maximum_path_length);
    }
    if (whole_path[index].maximum_duration_s > 0.0) {
      worst_quality_ratio = std::max(worst_quality_ratio,
          risk_local[index].maximum_duration_s /
              whole_path[index].maximum_duration_s);
    }
    if (index > 0) {
      whole_refined_p95_ms = std::max(
          whole_refined_p95_ms, whole_path[index].p95_backend_ms);
      local_refined_p95_ms = std::max(
          local_refined_p95_ms, risk_local[index].p95_backend_ms);
    }
  }
  const double p95_reduction = whole_refined_p95_ms > 0.0
      ? 1.0 - local_refined_p95_ms / whole_refined_p95_ms
      : 0.0;
  const bool accepted = all_success && p95_reduction >= 0.04 &&
      worst_quality_ratio <= 1.05;
  std::cout << std::fixed << std::setprecision(3)
            << "block_summary,all_success=" << all_success
            << ",whole_refined_p95_ms=" << whole_refined_p95_ms
            << ",risk_local_refined_p95_ms=" << local_refined_p95_ms
            << ",p95_reduction_percent=" << 100.0 * p95_reduction
            << ",worst_quality_ratio=" << worst_quality_ratio
            << ",accepted=" << accepted << '\n';
  return accepted ? 0 : 1;
}

int benchmarkNarrow(const std::filesystem::path& root) {
  const Request request{"narrow_regression", "narrow", {0.0, 0.0, 1.0},
                        -M_PI / 2.0, {9.0, 0.0, 1.0}, 0.0};
  const auto result = run(root, request, true);
  printResult(request, true, result);
  return result.success() ? 0 : 1;
}

bool acceptNarrowTEndpoint(const std::filesystem::path& root,
                           double goal_x,
                           const std::string& endpoint_name) {
  bool accepted = true;
  for (int index = 0; index < 12; ++index) {
    const int degrees = -180 + 30 * index;
    const Request request{
        "narrow_t_" + endpoint_name + "_goal_yaw_" +
            std::to_string(degrees),
        "narrow", {0.0, 0.0, 1.0}, 0.0, {goal_x, 0.0, 1.0},
        static_cast<double>(degrees) * M_PI / 180.0};
    const auto result = run(root, request, true);
    printResult(request, true, result);
    accepted = accepted && result.success() &&
        result.quality.path_length > 0.0 &&
        result.quality.duration_seconds > 0.0;
  }
  const Request free_yaw_request{
      "narrow_t_" + endpoint_name + "_goal_yaw_free", "narrow",
      {0.0, 0.0, 1.0}, 0.0, {goal_x, 0.0, 1.0}, 0.0, "t", false};
  const auto free_yaw_result = run(root, free_yaw_request, true);
  printResult(free_yaw_request, true, free_yaw_result);
  accepted = accepted && free_yaw_result.success() &&
      free_yaw_result.quality.path_length > 0.0 &&
      free_yaw_result.quality.duration_seconds > 0.0;
  return accepted;
}

int acceptNarrowT(const std::filesystem::path& root) {
  return acceptNarrowTEndpoint(root, 9.0, "right") ? 0 : 1;
}

int acceptNarrowTBidirectional(const std::filesystem::path& root) {
  const bool right_accepted = acceptNarrowTEndpoint(root, 9.0, "right");
  const bool left_accepted = acceptNarrowTEndpoint(root, -9.0, "left");
  return right_accepted && left_accepted ? 0 : 1;
}

int discoverBlock(const std::filesystem::path& root) {
  const std::array<Eigen::Vector3d, 5> starts{{
      {3.5, -2.5, 1.0}, {0.5, -0.5, 1.0}, {-4.5, 2.5, 1.0},
      {1.5, 3.5, 1.0}, {4.5, 1.5, 1.0},
  }};
  const std::array<Eigen::Vector3d, 10> goals{{
      {4.5, -2.5, 1.0}, {-1.5, 3.5, 1.0}, {-3.5, -4.5, 1.0},
      {0.5, -0.5, 1.0}, {-4.5, 2.5, 1.0}, {1.5, 3.5, 1.0},
      {4.5, 1.5, 1.0}, {6.5, 2.5, 1.0}, {-6.5, 2.5, 1.0},
      {-4.5, 5.5, 1.0},
  }};
  for (const auto& start : starts) {
    for (const auto& goal : goals) {
      Request candidate{"block_start_" + std::to_string(start.x()) + "_" +
              std::to_string(start.y()) + "_goal_" +
              std::to_string(goal.x()) + "_" + std::to_string(goal.y()),
          "block", start, 0.0, goal, 0.0};
      printResult(candidate, true, run(root, candidate, true));
    }
  }
  return 0;
}

int scanBlock(const std::filesystem::path& root, bool low_risk_first,
              bool acceptance_edge_first = false) {
  const auto scene = std::make_shared<const mighty::whole_body_safety::SafetyScene>(
      mighty::whole_body_safety::SafetyScene::load(
          (root / "maps/polymap/block.polymap.json").string(),
          (root / "shape/normal/ply/t.polymap.json").string(), 0.1, 3.0,
          1.15));
  mighty::whole_body_safety::ExponentialHingeOptions penalty;
  mighty::whole_body_safety::WholeBodySafety safety(scene, penalty);
  struct Candidate {
    double beta;
    Eigen::Vector3d position;
  };
  std::vector<Candidate> candidates;
  for (double x = -6.5; x <= 6.5; x += 1.0) {
    for (double y = -6.5; y <= 6.5; y += 1.0) {
      auto context = safety.createContext();
      const auto result = safety.evaluate({x, y, 1.0}, 0.0, context, true);
      if (result.success) {
        candidates.push_back({result.statistics.minimum_beta, {x, y, 1.0}});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [low_risk_first, acceptance_edge_first](const Candidate& left,
                                                     const Candidate& right) {
              if (acceptance_edge_first) {
                constexpr double kAcceptanceBeta = 0.9999;
                return std::abs(left.beta - kAcceptanceBeta) <
                    std::abs(right.beta - kAcceptanceBeta);
              }
              return low_risk_first ? left.beta < right.beta
                                    : left.beta > right.beta;
            });
  for (std::size_t index = 0; index < std::min<std::size_t>(20, candidates.size());
       ++index) {
    const auto& candidate = candidates[index];
    std::cout << std::fixed << std::setprecision(3)
              << "beta=" << candidate.beta << ",position="
              << candidate.position.transpose() << '\n';
  }
  return 0;
}

int runRequest(const std::filesystem::path& root, int argc, char** argv) {
  if (argc != 12 && argc != 13) return 2;
  const bool risk_local = argc == 12 || std::string(argv[12]) != "--baseline";
  Request request{"request", argv[2],
                  {std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5])},
                  std::stod(argv[6]),
                  {std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9])},
                  std::stod(argv[10])};
  if (std::string(argv[11]) != "--run") return 2;
  printResult(request, risk_local, run(root, request, risk_local));
  return 0;
}

int runOfflineRequest(const std::filesystem::path& root, int argc, char** argv) {
  bool free_start_yaw = false;
  bool risk_local = true;
  bool spatial_progress_yaw = true;
  bool risk_window = true;
  double start_yaw = 0.0;
  std::vector<std::string> positional;
  for (int index = 2; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--free-start-yaw") {
      free_start_yaw = true;
      continue;
    }
    if (arg == "--start-yaw-deg") {
      if (index + 1 >= argc) return 2;
      try {
        std::size_t consumed = 0;
        const double degrees = std::stod(argv[++index], &consumed);
        if (consumed != std::string(argv[index]).size() || !std::isfinite(degrees)) {
          return 2;
        }
        start_yaw = degrees * M_PI / 180.0;
      } catch (const std::exception&) {
        return 2;
      }
      if (!std::isfinite(start_yaw)) return 2;
      continue;
    }
    if (arg == "--method") {
      if (index + 1 >= argc) return 2;
      const std::string method = argv[++index];
      if (method == "ours") {
        risk_local = true;
        spatial_progress_yaw = true;
        risk_window = true;
      } else if (method == "whole_path") {
        risk_local = false;
        spatial_progress_yaw = true;
        risk_window = true;
      } else if (method == "time_phase_yaw") {
        risk_local = true;
        spatial_progress_yaw = false;
        risk_window = true;
      } else if (method == "no_risk_window") {
        risk_local = true;
        spatial_progress_yaw = true;
        risk_window = false;
      } else {
        return 2;
      }
      continue;
    }
    if (arg.rfind("--", 0) == 0) return 2;
    positional.push_back(arg);
  }
  if (positional.size() != 8 && positional.size() != 9) return 2;
  Request request{"offline", positional[0],
                  {std::stod(positional[2]), std::stod(positional[3]),
                   std::stod(positional[4])},
                  0.0,
                  {std::stod(positional[5]), std::stod(positional[6]),
                   std::stod(positional[7])},
                  positional.size() == 9 ? std::stod(positional[8]) : 0.0,
                  positional[1], positional.size() == 9};
  const auto map_path = root / "maps" / "polymap" / (request.map + ".polymap.json");
  const auto body_path = root / "shape" / (request.map == "hole" ? "small" : "normal") / "ply" /
                         (request.shape + ".polymap.json");
  MIGHTY planner(makeParameters(request.map, risk_local, spatial_progress_yaw,
                                risk_window),
                 makeFrontendConfig(request.map),
                 map_path.string(), body_path.string());
  state start{}; start.pos = request.start; start.yaw = start_yaw;
  if (free_start_yaw) {
    // This is a declared benchmark initial-condition choice, never an in-flight
    // pose correction. XYZ-only tasks may explicitly request a safe orientation.
    mighty::frontend::WholeBodyCollisionScene scene(
        makeFrontendConfig(request.map).collision, map_path.string(), body_path.string());
    if (!scene.isPoseCollisionFree({start.pos, start.yaw})) {
      const auto yaw = scene.nearestCollisionFreeYaw(start.pos, 0.0, 0.0);
      if (yaw) start.yaw = *yaw;
    }
  }
  planner.updateState(start);
  state goal{}; goal.pos = request.goal; goal.yaw = request.goal_yaw;
  goal.use_tracking_yaw = request.goal_yaw_exact;
  const auto result = planner.planGlobalOnce(goal, 0.0);
  std::vector<state> middle, backend; planner.retrieveMiddleGoalSetpoints(middle); planner.retrieveGoalSetpoints(backend);
  auto states = [](const std::vector<state>& source) { nlohmann::json out=nlohmann::json::array(); for (const auto& s:source) out.push_back({s.pos.x(),s.pos.y(),s.pos.z(),s.yaw,s.t}); return out; };
  nlohmann::json frontend=nlohmann::json::array();
  for (const auto& p:result.shape_path.waypoints) frontend.push_back({p.position.x(),p.position.y(),p.position.z(),p.yaw});
  nlohmann::json out={{"success",result.success()},{"reason",result.message},{"frontend",frontend},{"middle",states(middle)},{"backend",states(backend)},
    {"start_yaw",start.yaw},{"free_start_yaw",free_start_yaw},
    {"optimization_status",result.optimization_status},{"refinement_timed_out",result.refinement_timed_out},
    {"frontend_length",result.shape_path.statistics.path_length},{"frontend_ms",result.frontend_time_ms},{"middle_ms",result.middle_time_ms},{"backend_ms",result.backend_time_ms},{"duration_s",result.quality.duration_seconds},{"backend_length",result.quality.path_length},{"trajectory_yaw_variation",result.trajectory_yaw_variation},{"risk_segment_count",result.risk_segment_count},{"risk_segment_edges",result.risk_segment_edges},{"risk_window_count",result.risk_window_count},{"trajectory_segment_count",result.trajectory_segment_count}};
  // Preserve full setpoint states for faithful RViz replay of offline results.
  out["backend_samples"] = nlohmann::json::array();
  for (const auto& s : backend) {
    out["backend_samples"].push_back({{"t", s.t},
        {"p", {s.pos.x(), s.pos.y(), s.pos.z()}},
        {"v", {s.vel.x(), s.vel.y(), s.vel.z()}},
        {"a", {s.accel.x(), s.accel.y(), s.accel.z()}},
        {"q", {0.0, 0.0, std::sin(0.5 * s.yaw), std::cos(0.5 * s.yaw)}}});
  }
  std::cout << out.dump() << '\n'; return result.success() ? 0 : 1;
}

// Fixed experiment inputs and explicit map paths keep production assets intact.
int runPolyExperiment(const std::filesystem::path& root, int argc, char** argv) {
  if (argc != 7 && argc != 8)
    throw std::invalid_argument(
        "--poly-experiment manifest map-artifact output mode task-index [path.json]");
  std::ifstream input(argv[2]);
  const auto manifest = nlohmann::json::parse(input);
  const std::string map = manifest.at("map");
  const auto body = root / "shape" / (map == "hole" ? "small" : "normal") / "ply/t.polymap.json";
  auto config = makeFrontendConfig(map);
  std::ofstream output(argv[4]);
  if (!output) throw std::runtime_error("Cannot open result file");
  const auto vector = [](const nlohmann::json& v) {
    Eigen::Vector3d p(v.at(0).get<double>(),v.at(1).get<double>(),v.at(2).get<double>());
    if (!p.allFinite()) throw std::invalid_argument("Nonfinite position");
    return p;
  };
  const std::string mode = argv[5];
  if (mode == "collision") {
    mighty::frontend::WholeBodyCollisionScene scene(config.collision, argv[3], body.string());
    output << "query,free,time_us\n" << std::setprecision(17);
    std::size_t index = 0;
    for (const auto& value : manifest.at("queries")) {
      const mighty::frontend::PoseState pose{vector(value), value.at(3).get<double>()};
      const auto started = std::chrono::steady_clock::now();
      const bool free = scene.isPoseCollisionFree(pose); // Null context: no pose-result cache.
      const double elapsed = std::chrono::duration<double,std::micro>(
          std::chrono::steady_clock::now()-started).count();
      output << index++ << ',' << free << ',' << elapsed << '\n';
    }
  } else if (mode == "plan" || mode == "plan-stages" || mode == "plan-frontend" ||
             mode == "plan-middle" || mode == "plan-backend") {
    const bool isolated = mode == "plan-frontend" || mode == "plan-middle" ||
                          mode == "plan-backend";
    const bool stage_limits = mode == "plan-stages" || isolated;
    if ((mode == "plan-middle" || mode == "plan-backend") && argc != 8)
      throw std::invalid_argument("middle/backend isolation requires a certified path file");
    const auto& task = manifest.at("tasks").at(std::stoul(argv[6]));
    auto par = makeParameters(map,true);
    par.planner_Co = config.collision.body_clearance;
    if (stage_limits) {
      config.timeout_seconds = 1.0;
      par.global_once_refinement_timeout_seconds = 60.0;
      par.global_once_hard_timeout_seconds = 60.0;
    }
    MIGHTY planner(par, config, argv[3], body.string());
    state start{}; start.pos = vector(task.at("start")); start.yaw = 0;
    planner.updateState(start);
    state goal{}; goal.pos = vector(task.at("goal")); goal.use_tracking_yaw = false;
    // Only this experiment installs a process-local watchdog. Solver logic and
    // production timeout defaults are unchanged; retries share each stage budget.
    std::array<double,3> used{};
    const std::array<double,3> limits{1.0,1.0,5.0};
    int active = -1;
    auto started = std::chrono::steady_clock::now();
    const auto observe = [&](const char* stage) {
      itimerval timer{};
      setitimer(ITIMER_REAL,&timer,nullptr);
      const auto now = std::chrono::steady_clock::now();
      if (active >= 0) used[active] += std::chrono::duration<double>(now-started).count();
      active = std::string(stage)=="frontend" ? 0 : std::string(stage)=="middle" ? 1 : std::string(stage)=="backend" ? 2 : -1;
      started = now;
      std::cerr << "POLY_STAGE " << stage << ' ' << std::setprecision(17)
                << used[0] << ' ' << used[1] << ' ' << used[2] << '\n' << std::flush;
      if (active >= 0) {
        const double remaining = std::max(1.0e-6,limits[active]-used[active]);
        timer.it_value.tv_sec = static_cast<long>(remaining);
        timer.it_value.tv_usec = std::max(1L,static_cast<long>((remaining-timer.it_value.tv_sec)*1.0e6));
        if (setitimer(ITIMER_REAL,&timer,nullptr)) throw std::runtime_error("Cannot arm stage watchdog");
      }
    };
    if (stage_limits) {
      std::signal(SIGALRM, +[](int) {
        constexpr char message[]="POLY_TIMEOUT\n";
        (void)!write(STDERR_FILENO,message,sizeof(message)-1);
        _exit(124);
      });
      planner.setPlanningStageObserver(observe);
    }
    if (mode == "plan-frontend") planner.setPlanningStageLimit("frontend");
    else if (mode == "plan-middle") planner.setPlanningStageLimit("middle");
    else planner.setPlanningStageLimit(nullptr);
    GlobalPlanResult result;
    if (mode == "plan-middle" || mode == "plan-backend") {
      std::ifstream path_input(argv[7]);
      const auto path_json = nlohmann::json::parse(path_input);
      std::vector<mighty::frontend::PoseState> waypoints;
      for (const auto& value : path_json.at("waypoints")) {
        mighty::frontend::PoseState pose{vector(value), value.at(3).get<double>()};
        if (!std::isfinite(pose.yaw)) throw std::invalid_argument("Nonfinite yaw");
        waypoints.push_back(pose);
      }
      if (stage_limits) observe("idle");
      result = planner.planCertifiedShapePath(waypoints, 0.0);
    } else {
      if (stage_limits) observe("frontend");
      result = planner.planGlobalOnce(goal,0.0);
    }
    if (stage_limits) {
      observe("idle");
      planner.setPlanningStageObserver({});
      planner.setPlanningStageLimit(nullptr);
    }
    const auto optional_time = [](double t) -> nlohmann::json {
      return t < 0 ? nlohmann::json(nullptr) : nlohmann::json(t);
    };
    nlohmann::json waypoints = nlohmann::json::array();
    for (const auto& pose : result.shape_path.waypoints) {
      waypoints.push_back({pose.position.x(), pose.position.y(), pose.position.z(), pose.yaw});
    }
    const bool success = mode == "plan-frontend"
        ? result.shape_path.success() && result.shape_path.waypoints.size() >= 2
        : result.has_trajectory();
    if (stage_limits) output << "POLY_RESULT ";
    output << nlohmann::json({{"success",success}, {"reason",result.message},
      {"stage",static_cast<int>(result.stage)}, {"frontend_ms",result.frontend_time_ms},
      {"middle_ms",optional_time(result.seed_stage_time_ms)},
      {"backend_ms",optional_time(result.refinement_stage_time_ms)},
      {"refinement_timeout",result.refinement_timed_out},
      {"frontend_timeout",result.shape_path.status == mighty::frontend::PlanStatus::kTimedOut},
      {"path_length",result.shape_path.statistics.path_length},
      {"expansions",result.shape_path.statistics.expansions},
      {"collision_queries",result.shape_path.statistics.collision_queries},
      {"waypoints",waypoints}}).dump() << '\n';
  } else throw std::invalid_argument("Unknown experiment mode");
  if (!output) throw std::runtime_error("Result write failed");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path root =
      std::filesystem::path(MIGHTY_SOURCE_DIR);
  if (argc >= 2 && std::string(argv[1]) == "--poly-experiment") {
    try { return runPolyExperiment(root,argc,argv); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
  }
  if (argc == 2 && std::string(argv[1]) == "--discover-block") {
    return discoverBlock(root);
  }
  if (argc == 2 && std::string(argv[1]) == "--benchmark-block") {
    return benchmarkBlock(root);
  }
  if (argc == 2 && std::string(argv[1]) == "--benchmark-narrow") {
    return benchmarkNarrow(root);
  }
  if (argc == 2 && std::string(argv[1]) == "--acceptance-narrow-t") {
    return acceptNarrowT(root);
  }
  if (argc == 2 &&
      std::string(argv[1]) == "--acceptance-narrow-t-bidirectional") {
    return acceptNarrowTBidirectional(root);
  }
  if (argc == 2 && std::string(argv[1]) == "--scan-block") {
    return scanBlock(root, false);
  }
  if (argc == 2 && std::string(argv[1]) == "--scan-block-low") {
    return scanBlock(root, true);
  }
  if (argc == 2 && std::string(argv[1]) == "--scan-block-edge") {
    return scanBlock(root, false, true);
  }
  if (argc >= 2 && std::string(argv[1]) == "--request") {
    return runRequest(root, argc, argv);
  }
  if (argc >= 2 && std::string(argv[1]) == "--offline-request") return runOfflineRequest(root, argc, argv);
  std::cerr << "usage: benchmark_risk_local_backend --benchmark-block|"
               "--benchmark-narrow|"
               "--acceptance-narrow-t|"
               "--acceptance-narrow-t-bidirectional|"
               "--discover-block|--scan-block|"
               "--scan-block-low|--scan-block-edge\n"
            << "   or: benchmark_risk_local_backend --request <map> "
               "<sx> <sy> <sz> <syaw> <gx> <gy> <gz> <gyaw> --run "
               "[--baseline]\n";
  return 2;
}
