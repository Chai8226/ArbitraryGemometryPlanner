/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <algorithm>  // for std::clamp
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include <mighty/lbfgs_solver_utils.hpp>
#include "mighty/eigen_types.hpp"

struct StateDeriv {
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
  Eigen::Vector3d accel;
  Eigen::Vector3d jerk;
};

struct polytope {
  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
};

struct parameters {
  // Sim enviroment
  std::string sim_env;
  std::string planning_mode{"global_once"};
  std::string map_mode{"known_pcd"};
  std::string known_map_pcd_path;
  std::string known_map_polymap_path;
  double known_map_resolution{0.1};
  double global_once_refinement_timeout_seconds{3.0};
  double global_once_hard_timeout_seconds{5.0};
  double global_once_minimum_interior_speed{0.2};
  std::string init_mode{"topp"};
  bool light_activation_ladder{true};
  // Test-only baseline selection.  Production uses the default risk-local
  // policy and does not expose a ROS parameter for this switch.
  bool risk_local_refinement{true};
  // Test-only yaw-reference baseline.  Production couples yaw to spatial
  // progress and does not expose a ROS parameter for this switch.
  bool spatial_progress_yaw_coupling{true};
  // Test-only: skip overlapping windows and jointly solve the risk mask.
  // Production always uses risk-window refinement on long risk blocks.
  bool risk_window_refinement{true};
  double planner_Co_normal{0.1};
  double planner_Co_small{0.05};
  double planner_Co_real{0.05};

  std::string map_frame_id{"map"};
  // Visual level
  int visual_level;

  // Global planner parameters
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double z_min;
  double z_max;
  double drone_radius;
  bool use_state_update;

  // Trajectory segmentation
  double max_dist_vertexes;  // [m] Maximum distance between two consecutive vertexes

  // Optimization parameters
  double dc;
  double v_max;
  double a_max;
  double j_max;
  double jerk_weight;
  double yaw_jerk_weight{0.1};
  double yaw_reference_weight{0.0};
  double time_weight;
  double pos_anchor_weight;
  double stat_weight;
  double dyn_constr_bodyrate_weight;
  double dyn_constr_tilt_weight;
  double dyn_constr_thrust_weight;
  double dyn_constr_vel_weight;
  double dyn_constr_acc_weight;
  double dyn_constr_jerk_weight;
  double planner_Co;         // static clearance; whole-body mode inflates Shape once
  double safety_beta_soft{1.15};
  double safety_exponential_gain{6.0};
  double safety_hard_weight_multiplier{20.0};
  double safety_position_sample_step{0.2};
  double safety_yaw_sample_step{15.0 * M_PI / 180.0};
  double safety_ipm_tolerance{1.0e-7};
  int safety_ipm_warm_iterations{12};
  int safety_ipm_cold_iterations{30};
  double safety_ipm_regularization{1.0e-9};
  double safety_acceptance_beta{1.0 - 1.0e-4};
  std::vector<double> drone_bbox;
  double goal_radius;
  int spline_degree{5};     // [-] Hermite spline degree: 3 (cubic) or 5 (quintic)

  // L-BFGS parameters
  double f_dec_coeff;      // allow larger Armijo steps
  double cautious_factor;  // always accept BFGS update
  int past;                // number of past iterations to use
  int max_linesearch;      // fewer backtracking tries
  int max_iterations;      // allow more iterations
  double g_epsilon;        // gradient norm threshold for convergence
  double delta;            // stop once f-improvement is minimal

  // Yaw-related parameters
  double alpha_filter_dyaw;
  double w_max;
  double yaw_acc_max{2.0};
  double yaw_goal_tolerance{0.04};
  int yaw_spinning_threshold;
  double yaw_spinning_dyaw;

  // Simulation env parameters
  bool force_goal_z;
  double default_goal_z;

  // Debug flags
  bool debug_verbose;

  // Trajectory publishing parameters
  int trajectory_downsample_points{500};  // Number of points to downsample trajectory to
  double mpc_path_spacing{0.05};          // [m] Spacing between waypoints in MPC path

};

struct BasisConverter {
  Eigen::Matrix<double, 4, 4> A_pos_mv_rest;
  Eigen::Matrix<double, 4, 4> A_pos_mv_rest_inv;
  Eigen::Matrix<double, 3, 3> A_vel_mv_rest;
  Eigen::Matrix<double, 3, 3> A_vel_mv_rest_inv;
  Eigen::Matrix<double, 2, 2> A_accel_mv_rest;
  Eigen::Matrix<double, 2, 2> A_accel_mv_rest_inv;
  Eigen::Matrix<double, 4, 4> A_pos_be_rest;
  Eigen::Matrix<double, 4, 4> A_pos_bs_seg0, A_pos_bs_seg1, A_pos_bs_rest, A_pos_bs_seg_last2,
      A_pos_bs_seg_last;

  Eigen::Matrix<double, 4, 4> M_pos_bs2mv_seg0, M_pos_bs2mv_seg1, M_pos_bs2mv_rest,
      M_pos_bs2mv_seg_last2, M_pos_bs2mv_seg_last;

  Eigen::Matrix<double, 4, 4> M_pos_bs2be_seg0, M_pos_bs2be_seg1, M_pos_bs2be_rest,
      M_pos_bs2be_seg_last2, M_pos_bs2be_seg_last;

  Eigen::Matrix<double, 3, 3> M_vel_bs2mv_seg0, M_vel_bs2mv_rest, M_vel_bs2mv_seg_last;
  Eigen::Matrix<double, 3, 3> M_vel_bs2be_seg0, M_vel_bs2be_rest, M_vel_bs2be_seg_last;

  BasisConverter() {
    // See matlab.
    // This is for t \in [0 1];

    //////MATRICES A FOR MINVO POSITION///////// (there is only one)
    A_pos_mv_rest << -3.4416308968564117698463178385282, 6.9895481477801393310755884158425,
        -4.4622887507045296828778191411402, 0.91437149978080234369315348885721,
        6.6792587327074839365081970754545, -11.845989901556746914934592496138,
        5.2523596690684613008670567069203, 0, -6.6792587327074839365081970754545,
        8.1917862965657040064115790301003, -1.5981560640774179482548333908198,
        0.085628500219197656306846511142794, 3.4416308968564117698463178385282,
        -3.3353445427890959784633650997421, 0.80808514571348655231020075007109,
        -0.0000000000000000084567769453869345852581318467855;

    //////INVERSE OF A_pos_mv_rest
    A_pos_mv_rest_inv = A_pos_mv_rest.inverse();

    //////MATRICES A FOR MINVO VELOCITY///////// (there is only one)
    A_vel_mv_rest << 1.4999999992328318931811281800037, -2.3660254034601951866889635311964,
        0.9330127021136816189983420599674, -2.9999999984656637863622563600074,
        2.9999999984656637863622563600074, 0, 1.4999999992328318931811281800037,
        -0.6339745950054685996732928288111, 0.066987297886318325490506708774774;

    //////INVERSE OF A_vel_mv_rest
    A_vel_mv_rest_inv = A_vel_mv_rest.inverse();

    //////MATRICES A FOR MINVO ACCELERATION///////// (there is only one)
    A_accel_mv_rest << -1.0, 1.0, 1.0, 0.0;

    /////INVERSE OF A_accel_mv_rest
    A_accel_mv_rest_inv = A_accel_mv_rest.inverse();

    //////MATRICES A FOR Bezier POSITION///////// (there is only one)
    A_pos_be_rest <<

        -1.0,
        3.0, -3.0, 1.0, 3.0, -6.0, 3.0, 0, -3.0, 3.0, 0, 0, 1.0, 0, 0, 0;

    //////MATRICES A FOR BSPLINE POSITION/////////
    A_pos_bs_seg0 <<

        -1.0000,
        3.0000, -3.0000, 1.0000, 1.7500, -4.5000, 3.0000, 0, -0.9167, 1.5000, 0, 0, 0.1667, 0, 0, 0;

    A_pos_bs_seg1 <<

        -0.2500,
        0.7500, -0.7500, 0.2500, 0.5833, -1.2500, 0.2500, 0.5833, -0.5000, 0.5000, 0.5000, 0.1667,
        0.1667, 0, 0, 0;

    A_pos_bs_rest <<

        -0.1667,
        0.5000, -0.5000, 0.1667, 0.5000, -1.0000, 0, 0.6667, -0.5000, 0.5000, 0.5000, 0.1667,
        0.1667, 0, 0, 0;

    A_pos_bs_seg_last2 <<

        -0.1667,
        0.5000, -0.5000, 0.1667, 0.5000, -1.0000, 0.0000, 0.6667, -0.5833, 0.5000, 0.5000, 0.1667,
        0.2500, 0, 0, 0;

    A_pos_bs_seg_last <<

        -0.1667,
        0.5000, -0.5000, 0.1667, 0.9167, -1.2500, -0.2500, 0.5833, -1.7500, 0.7500, 0.7500, 0.2500,
        1.0000, 0, 0, 0;

    //////BSPLINE to MINVO POSITION/////////

    M_pos_bs2mv_seg0 <<

        1.1023313949144333268037598827505,
        0.34205724556666972091534262290224, -0.092730934245582874453361910127569,
        -0.032032766697130621302846975595457, -0.049683556253749178166501110354147,
        0.65780347324677179710050722860615, 0.53053863760186903419935333658941,
        0.21181027098212013015654520131648, -0.047309044211162346038612724896666,
        0.015594436894155586093013710069499, 0.5051827557159349613158383363043,
        0.63650059656260427054519368539331, -0.0053387944495217444854096022766043,
        -0.015455155707597083292181849856206, 0.057009540927778303009976212933907,
        0.18372189915240558222286892942066;

    M_pos_bs2mv_seg1 <<

        0.27558284872860833170093997068761,
        0.085514311391667430228835655725561, -0.023182733561395718613340477531892,
        -0.0080081916742826553257117438988644, 0.6099042761975865811763242163579,
        0.63806904207840509091198555324809, 0.29959938009132258684985572472215,
        0.12252106674808682651445224109921, 0.11985166952332682033244282138185,
        0.29187180223752445806795208227413, 0.66657381254229419731416328431806,
        0.70176522577378930289881964199594, -0.0053387944495217444854096022766043,
        -0.015455155707597083292181849856206, 0.057009540927778303009976212933907,
        0.18372189915240558222286892942066;

    M_pos_bs2mv_rest <<

        0.18372189915240555446729331379174,
        0.057009540927778309948870116841135, -0.015455155707597117986651369392348,
        -0.0053387944495218164764338553140988, 0.70176522577378919187651717948029,
        0.66657381254229419731416328431806, 0.29187180223752384744528853843804,
        0.11985166952332582113172065874096, 0.11985166952332682033244282138185,
        0.29187180223752445806795208227413, 0.66657381254229419731416328431806,
        0.70176522577378930289881964199594, -0.0053387944495217444854096022766043,
        -0.015455155707597083292181849856206, 0.057009540927778303009976212933907,
        0.18372189915240558222286892942066;

    M_pos_bs2mv_seg_last2 <<

        0.18372189915240569324517139193631,
        0.057009540927778309948870116841135, -0.015455155707597145742226985021261,
        -0.0053387944495218164764338553140988, 0.70176522577378952494342456702725,
        0.66657381254229453038107067186502, 0.29187180223752412500104469472717,
        0.11985166952332593215402312125661, 0.1225210667480875342816304396365,
        0.29959938009132280889446064975346, 0.63806904207840497988968309073243,
        0.60990427619758624810941682881094, -0.0080081916742826154270717964323012,
        -0.023182733561395621468825822830695, 0.085514311391667444106623463540018,
        0.27558284872860833170093997068761;

    M_pos_bs2mv_seg_last <<

        0.18372189915240555446729331379174,
        0.057009540927778309948870116841135, -0.015455155707597117986651369392348,
        -0.0053387944495218164764338553140988, 0.63650059656260415952289122287766,
        0.5051827557159349613158383363043, 0.015594436894155294659469745965907,
        -0.047309044211162887272337229660479, 0.21181027098212068526805751389475,
        0.53053863760186914522165579910506, 0.65780347324677146403359984105919,
        -0.049683556253749622255710960416764, -0.032032766697130461708287185729205,
        -0.09273093424558248587530329132278, 0.34205724556666977642649385416007,
        1.1023313949144333268037598827505;

    //////BSPLINE to BEZIER POSITION/////////

    M_pos_bs2be_seg0 <<

        1.0000,
        0.0000, -0.0000, 0, 0, 1.0000, 0.5000, 0.2500, 0, -0.0000, 0.5000, 0.5833, 0, 0, 0, 0.1667;

    M_pos_bs2be_seg1 <<

        0.2500,
        0.0000, -0.0000, 0, 0.5833, 0.6667, 0.3333, 0.1667, 0.1667, 0.3333, 0.6667, 0.6667, 0, 0, 0,
        0.1667;

    M_pos_bs2be_rest <<

        0.1667,
        0.0000, 0, 0, 0.6667, 0.6667, 0.3333, 0.1667, 0.1667, 0.3333, 0.6667, 0.6667, 0, 0, 0,
        0.1667;

    M_pos_bs2be_seg_last2 <<

        0.1667,
        0, -0.0000, 0, 0.6667, 0.6667, 0.3333, 0.1667, 0.1667, 0.3333, 0.6667, 0.5833, 0, 0, 0,
        0.2500;

    M_pos_bs2be_seg_last <<

        0.1667,
        0.0000, 0, 0, 0.5833, 0.5000, 0, 0, 0.2500, 0.5000, 1.0000, 0, 0, 0, 0, 1.0000;

    /////BSPLINE to MINVO VELOCITY
    M_vel_bs2mv_seg0 <<

        1.077349059083916,
        0.1666702138890985, -0.07735049175615138, -0.03867488648729411, 0.7499977187062712,
        0.5386802643920123, -0.03867417280506149, 0.08333206631563977, 0.538670227146185;

    M_vel_bs2mv_rest <<

        0.538674529541958,
        0.08333510694454926, -0.03867524587807569, 0.4999996430546639, 0.8333328256508203,
        0.5000050185139366, -0.03867417280506149, 0.08333206631563977, 0.538670227146185;

    M_vel_bs2mv_seg_last <<

        0.538674529541958,
        0.08333510694454926, -0.03867524587807569, 0.5386738158597254, 0.7500007593351806,
        -0.03866520863224832, -0.07734834561012298, 0.1666641326312795, 1.07734045429237;

    /////BSPLINE to BEZIER VELOCITY
    M_vel_bs2be_seg0 <<

        1.0000,
        0, 0, 0, 1.0000, 0.5000, 0, 0, 0.5000;

    M_vel_bs2be_rest <<

        0.5000,
        0, 0, 0.5000, 1.0000, 0.5000, 0, 0, 0.5000;

    M_vel_bs2be_seg_last <<

        0.5000,
        0, 0, 0.5000, 1.0000, 0, 0, 0, 1.0000;
  }

  //////MATRIX A FOR MINVO POSITION/////////
  Eigen::Matrix<double, 4, 4> getArestMinvo() { return A_pos_mv_rest; }
  //////MATRIX A FOR Bezier POSITION/////////
  Eigen::Matrix<double, 4, 4> getArestBezier() { return A_pos_be_rest; }

  //////MATRIX A FOR BSPLINE POSITION/////////
  Eigen::Matrix<double, 4, 4> getArestBSpline() { return A_pos_bs_rest; }

  //////MATRICES A FOR MINVO POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getAMinvo(int num_pol) {
    std::vector<Eigen::Matrix<double, 4, 4>> A_pos_mv;  // will have as many elements as num_pol
    for (int i = 0; i < num_pol; i++) {
      A_pos_mv.push_back(A_pos_mv_rest);
    }
    return A_pos_mv;
  }

  //////MATRICES A FOR Bezier POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getABezier(int num_pol) {
    std::vector<Eigen::Matrix<double, 4, 4>> A_pos_be;  // will have as many elements as num_pol
    for (int i = 0; i < num_pol; i++) {
      A_pos_be.push_back(A_pos_be_rest);
    }
    return A_pos_be;
  }

  //////MATRICES A FOR BSPLINE POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getABSpline(int num_pol) {
    std::vector<Eigen::Matrix<double, 4, 4>> A_pos_bs;  // will have as many elements as num_pol
    A_pos_bs.push_back(A_pos_bs_seg0);
    A_pos_bs.push_back(A_pos_bs_seg1);
    for (int i = 0; i < (num_pol - 4); i++) {
      A_pos_bs.push_back(A_pos_bs_rest);
    }
    A_pos_bs.push_back(A_pos_bs_seg_last2);
    A_pos_bs.push_back(A_pos_bs_seg_last);
    return A_pos_bs;
  }

  //////BSPLINE to MINVO POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getMinvoPosConverters(int num_pol) {
    std::vector<Eigen::Matrix<double, 4, 4>> M_pos_bs2mv;  // will have as many elements as num_pol
    M_pos_bs2mv.push_back(M_pos_bs2mv_seg0);
    M_pos_bs2mv.push_back(M_pos_bs2mv_seg1);
    for (int i = 0; i < (num_pol - 4); i++) {
      M_pos_bs2mv.push_back(M_pos_bs2mv_rest);
    }
    M_pos_bs2mv.push_back(M_pos_bs2mv_seg_last2);
    M_pos_bs2mv.push_back(M_pos_bs2mv_seg_last);
    return M_pos_bs2mv;
  }

  //////BEZIER to MINVO POSITION/////////
  //////Q_{MINVO} = M_{BEZIER2MINVO} * Q_{BEZIER}
  Eigen::Matrix<double, 4, 4> getMinvoPosConverterFromBezier() {
    // Compute the conversion matrix for one segment
    Eigen::Matrix<double, 4, 4> M_be2mv = A_pos_mv_rest_inv * A_pos_be_rest;

    return M_be2mv;
  }

  //////BSPLINE to BEZIER POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getBezierPosConverters(int num_pol) {
    std::vector<Eigen::Matrix<double, 4, 4>> M_pos_bs2be;  // will have as many elements as num_pol
    M_pos_bs2be.push_back(M_pos_bs2be_seg0);
    M_pos_bs2be.push_back(M_pos_bs2be_seg1);
    for (int i = 0; i < (num_pol - 4); i++) {
      M_pos_bs2be.push_back(M_pos_bs2be_rest);
    }
    M_pos_bs2be.push_back(M_pos_bs2be_seg_last2);
    M_pos_bs2be.push_back(M_pos_bs2be_seg_last);
    return M_pos_bs2be;
  }

  //////BSPLINE to BSPLINE POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getBSplinePosConverters(int num_pol) {
    std::vector<Eigen::Matrix<double, 4, 4>> M_pos_bs2bs;  // will have as many elements as num_pol
    for (int i = 0; i < num_pol; i++) {
      M_pos_bs2bs.push_back(Eigen::Matrix<double, 4, 4>::Identity());
    }
    return M_pos_bs2bs;
  }

  //////BSPLINE to MINVO Velocity/////////
  std::vector<Eigen::Matrix<double, 3, 3>> getMinvoVelConverters(int num_pol) {
    std::vector<Eigen::Matrix<double, 3, 3>> M_vel_bs2mv;  // will have as many elements as num_pol
    M_vel_bs2mv.push_back(M_vel_bs2mv_seg0);
    for (int i = 0; i < (num_pol - 2 - 1); i++) {
      M_vel_bs2mv.push_back(M_vel_bs2mv_rest);
    }
    M_vel_bs2mv.push_back(M_vel_bs2mv_seg_last);
    return M_vel_bs2mv;
  }

  //////BSPLINE to BEZIER Velocity/////////
  std::vector<Eigen::Matrix<double, 3, 3>> getBezierVelConverters(int num_pol) {
    std::vector<Eigen::Matrix<double, 3, 3>> M_vel_bs2be;  // will have as many elements as segments
    M_vel_bs2be.push_back(M_vel_bs2be_seg0);
    for (int i = 0; i < (num_pol - 2 - 1); i++) {
      M_vel_bs2be.push_back(M_vel_bs2be_rest);
    }
    M_vel_bs2be.push_back(M_vel_bs2be_seg_last);
    return M_vel_bs2be;
  }

  //////BSPLINE to BSPLINE Velocity/////////
  std::vector<Eigen::Matrix<double, 3, 3>> getBSplineVelConverters(int num_pol) {
    std::vector<Eigen::Matrix<double, 3, 3>> M_vel_bs2bs;  // will have as many elements as num_pol
    for (int i = 0; i < num_pol; i++) {
      M_vel_bs2bs.push_back(Eigen::Matrix<double, 3, 3>::Identity());
    }
    return M_vel_bs2bs;
  }
};

struct PieceWisePol {
  // Interval 0: t\in[t0, t1)
  // Interval 1: t\in[t1, t2)
  // Interval 2: t\in[t2, t3)
  //...
  // Interval n-1: t\in[tn, tn+1)

  // n intervals in total

  // times has n+1 elements
  std::vector<double> times;  // [t0,t1,t2,...,tn+1]

  // coefficients has n elements
  // The coeffients are such that pol(t)=coeff_of_that_interval*[u^3 u^2 u 1]
  // with u=(t-t_min_that_interval)/(t_max_that_interval- t_min_that_interval)
  std::vector<Eigen::Matrix<double, 4, 1>> coeff_x;  // [a b c d]' of Int0 , [a b c d]' of Int1,...
  std::vector<Eigen::Matrix<double, 4, 1>> coeff_y;  // [a b c d]' of Int0 , [a b c d]' of Int1,...
  std::vector<Eigen::Matrix<double, 4, 1>> coeff_z;  // [a b c d]' of Int0 , [a b c d]' of Int1,...

  void clear() {
    times.clear();
    coeff_x.clear();
    coeff_y.clear();
    coeff_z.clear();
  }

  // Get the end time of the trajectory
  double getEndTime() const { return times.back(); }

  inline Eigen::Vector3d eval(double t) const {
    Eigen::Vector3d result;

    // return the last value of the polynomial in the last interval
    if (t >= times.back()) {
      Eigen::Matrix<double, 4, 1> tmp;
      // double u = 1;
      double u = times.back() - times[times.size() - 2];
      tmp << u * u * u, u * u, u, 1.0;
      result.x() = coeff_x.back().transpose() * tmp;
      result.y() = coeff_y.back().transpose() * tmp;
      result.z() = coeff_z.back().transpose() * tmp;
      return result;
    }

    // return the first value of the polynomial in the first interval
    if (t < times.front()) {
      Eigen::Matrix<double, 4, 1> tmp;
      double u = 0;
      tmp << u * u * u, u * u, u, 1.0;
      result.x() = coeff_x.front().transpose() * tmp;
      result.y() = coeff_y.front().transpose() * tmp;
      result.z() = coeff_z.front().transpose() * tmp;
      return result;
    }

    // Find the interval where t is
    //(times - 1) is the number of intervals
    for (int i = 0; i < (times.size() - 1); i++) {
      if (times[i] <= t && t < times[i + 1]) {
        // double u = (t - times[i]) / (times[i + 1] - times[i]);
        double u = t - times[i];

        Eigen::Matrix<double, 4, 1> tmp;
        tmp << u * u * u, u * u, u, 1.0;

        result.x() = coeff_x[i].transpose() * tmp;
        result.y() = coeff_y[i].transpose() * tmp;
        result.z() = coeff_z[i].transpose() * tmp;

        break;
      }
    }
    return result;
  }

  inline Eigen::Vector3d velocity(double t) const {
    Eigen::Vector3d vel;

    // Handle the case where t is after the last interval
    if (t >= times.back()) {
      double u = times.back() - times[times.size() - 2];
      vel.x() = 3 * coeff_x.back()(0) * u * u + 2 * coeff_x.back()(1) * u + coeff_x.back()(2);
      vel.y() = 3 * coeff_y.back()(0) * u * u + 2 * coeff_y.back()(1) * u + coeff_y.back()(2);
      vel.z() = 3 * coeff_z.back()(0) * u * u + 2 * coeff_z.back()(1) * u + coeff_z.back()(2);
      return vel;
    }

    // Handle the case where t is before the first interval
    if (t < times.front()) {
      vel.x() = coeff_x.front()(2);
      vel.y() = coeff_y.front()(2);
      vel.z() = coeff_z.front()(2);
      return vel;
    }

    // Find the interval where t lies and calculate velocity
    for (int i = 0; i < (times.size() - 1); i++) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        vel.x() = 3 * coeff_x[i](0) * u * u + 2 * coeff_x[i](1) * u + coeff_x[i](2);
        vel.y() = 3 * coeff_y[i](0) * u * u + 2 * coeff_y[i](1) * u + coeff_y[i](2);
        vel.z() = 3 * coeff_z[i](0) * u * u + 2 * coeff_z[i](1) * u + coeff_z[i](2);
        break;
      }
    }

    return vel;
  }

  /// Evaluate acceleration (second derivative) at time t
  inline Eigen::Vector3d acceleration(double t) const {
    Eigen::Vector3d a{0, 0, 0};

    // 1) If t is after the last interval:
    if (t >= times.back()) {
      double u = times.back() - times[times.size() - 2];
      // coeff_* .back()(0) is the cubic a, .back()(1) is the quadratic b
      a.x() = 6.0 * coeff_x.back()(0) * u + 2.0 * coeff_x.back()(1);
      a.y() = 6.0 * coeff_y.back()(0) * u + 2.0 * coeff_y.back()(1);
      a.z() = 6.0 * coeff_z.back()(0) * u + 2.0 * coeff_z.back()(1);
      return a;
    }

    // 2) If t is before the first interval:
    if (t < times.front()) {
      a.x() = 2.0 * coeff_x.front()(1);
      a.y() = 2.0 * coeff_y.front()(1);
      a.z() = 2.0 * coeff_z.front()(1);
      return a;
    }

    // 3) Otherwise, find the correct interval i
    for (size_t i = 0; i + 1 < times.size(); ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        a.x() = 6.0 * coeff_x[i](0) * u + 2.0 * coeff_x[i](1);
        a.y() = 6.0 * coeff_y[i](0) * u + 2.0 * coeff_y[i](1);
        a.z() = 6.0 * coeff_z[i](0) * u + 2.0 * coeff_z[i](1);
        break;
      }
    }
    return a;
  }

  void print() const {
    std::cout << "coeff_x.size()= " << coeff_x.size() << std::endl;
    std::cout << "times.size()= " << times.size() << std::endl;
    std::cout << "Note that coeff_x.size() == times.size()-1" << std::endl;

    for (int i = 0; i < times.size(); i++) {
      printf("Time: %f\n", times[i]);
    }

    for (int i = 0; i < (times.size() - 1); i++) {
      std::cout << "From " << times[i] << " to " << times[i + 1] << std::endl;
      std::cout << "  Coeff_x= " << coeff_x[i].transpose() << std::endl;
      std::cout << "  Coeff_y= " << coeff_y[i].transpose() << std::endl;
      std::cout << "  Coeff_z= " << coeff_z[i].transpose() << std::endl;
    }
  }
};

struct PieceWiseQuinticPol {
  // breakpoints [t0, t1, 鈥?, tN]
  std::vector<double> times;

  // per鈥恠egment coefficients [a b c d e f]岬� for a路u鈦? + b路u鈦? + c路u鲁 + d路u虏 + e路u + f
  std::vector<Eigen::Matrix<double, 6, 1>> coeff_x, coeff_y, coeff_z;

  /// Remove all intervals
  void clear() {
    times.clear();
    coeff_x.clear();
    coeff_y.clear();
    coeff_z.clear();
  }

  /// End time of the trajectory
  double getEndTime() const { return times.back(); }

  /// Position p = a路u鈦? + b路u鈦? + c路u鲁 + d路u虏 + e路u + f
  inline Eigen::Vector3d eval(double t) const {
    Eigen::Vector3d r{0, 0, 0};

    // (1) after last: clamp to end of last interval
    if (t >= times.back()) {
      double u = times.back() - times[times.size() - 2];
      Eigen::Matrix<double, 6, 1> U;
      U << u * u * u * u * u, u * u * u * u, u * u * u, u * u, u, 1.0;
      r.x() = coeff_x.back().dot(U);
      r.y() = coeff_y.back().dot(U);
      r.z() = coeff_z.back().dot(U);
      return r;
    }

    // (2) before first: u=0
    if (t < times.front()) {
      Eigen::Matrix<double, 6, 1> U;
      U << 0, 0, 0, 0, 0, 1.0;
      r.x() = coeff_x.front().dot(U);
      r.y() = coeff_y.front().dot(U);
      r.z() = coeff_z.front().dot(U);
      return r;
    }

    // (3) somewhere in the middle
    for (size_t i = 0; i + 1 < times.size(); ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        Eigen::Matrix<double, 6, 1> U;
        U << u * u * u * u * u, u * u * u * u, u * u * u, u * u, u, 1.0;
        r.x() = coeff_x[i].dot(U);
        r.y() = coeff_y[i].dot(U);
        r.z() = coeff_z[i].dot(U);
        break;
      }
    }
    return r;
  }

  /// Velocity p鈥? = 5a路u鈦? + 4b路u鲁 + 3c路u虏 + 2d路u + e
  inline Eigen::Vector3d velocity(double t) const {
    Eigen::Vector3d v{0, 0, 0};

    if (t >= times.back()) {
      double u = times.back() - times[times.size() - 2];
      v.x() = 5 * coeff_x.back()(0) * u * u * u * u + 4 * coeff_x.back()(1) * u * u * u +
              3 * coeff_x.back()(2) * u * u + 2 * coeff_x.back()(3) * u + coeff_x.back()(4);
      v.y() = 5 * coeff_y.back()(0) * u * u * u * u + 4 * coeff_y.back()(1) * u * u * u +
              3 * coeff_y.back()(2) * u * u + 2 * coeff_y.back()(3) * u + coeff_y.back()(4);
      v.z() = 5 * coeff_z.back()(0) * u * u * u * u + 4 * coeff_z.back()(1) * u * u * u +
              3 * coeff_z.back()(2) * u * u + 2 * coeff_z.back()(3) * u + coeff_z.back()(4);
      return v;
    }

    if (t < times.front()) {
      v.x() = coeff_x.front()(4);
      v.y() = coeff_y.front()(4);
      v.z() = coeff_z.front()(4);
      return v;
    }

    for (size_t i = 0; i + 1 < times.size(); ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        v.x() = 5 * coeff_x[i](0) * u * u * u * u + 4 * coeff_x[i](1) * u * u * u +
                3 * coeff_x[i](2) * u * u + 2 * coeff_x[i](3) * u + coeff_x[i](4);
        v.y() = 5 * coeff_y[i](0) * u * u * u * u + 4 * coeff_y[i](1) * u * u * u +
                3 * coeff_y[i](2) * u * u + 2 * coeff_y[i](3) * u + coeff_y[i](4);
        v.z() = 5 * coeff_z[i](0) * u * u * u * u + 4 * coeff_z[i](1) * u * u * u +
                3 * coeff_z[i](2) * u * u + 2 * coeff_z[i](3) * u + coeff_z[i](4);
        break;
      }
    }
    return v;
  }

  /// Acceleration p鈥? = 20a路u鲁 + 12b路u虏 + 6c路u + 2d
  inline Eigen::Vector3d acceleration(double t) const {
    Eigen::Vector3d a{0, 0, 0};

    if (t >= times.back()) {
      double u = times.back() - times[times.size() - 2];
      a.x() = 20 * coeff_x.back()(0) * u * u * u + 12 * coeff_x.back()(1) * u * u +
              6 * coeff_x.back()(2) * u + 2 * coeff_x.back()(3);
      a.y() = 20 * coeff_y.back()(0) * u * u * u + 12 * coeff_y.back()(1) * u * u +
              6 * coeff_y.back()(2) * u + 2 * coeff_y.back()(3);
      a.z() = 20 * coeff_z.back()(0) * u * u * u + 12 * coeff_z.back()(1) * u * u +
              6 * coeff_z.back()(2) * u + 2 * coeff_z.back()(3);
      return a;
    }

    if (t < times.front()) {
      a.x() = 2 * coeff_x.front()(3);
      a.y() = 2 * coeff_y.front()(3);
      a.z() = 2 * coeff_z.front()(3);
      return a;
    }

    for (size_t i = 0; i + 1 < times.size(); ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        a.x() = 20 * coeff_x[i](0) * u * u * u + 12 * coeff_x[i](1) * u * u +
                6 * coeff_x[i](2) * u + 2 * coeff_x[i](3);
        a.y() = 20 * coeff_y[i](0) * u * u * u + 12 * coeff_y[i](1) * u * u +
                6 * coeff_y[i](2) * u + 2 * coeff_y[i](3);
        a.z() = 20 * coeff_z[i](0) * u * u * u + 12 * coeff_z[i](1) * u * u +
                6 * coeff_z[i](2) * u + 2 * coeff_z[i](3);
        break;
      }
    }
    return a;
  }

  /// Jerk p鈥? = 60a路u虏 + 24b路u + 6c
  inline Eigen::Vector3d jerk(double t) const {
    Eigen::Vector3d j{0, 0, 0};

    if (t >= times.back()) {
      double u = times.back() - times[times.size() - 2];
      j.x() = 60 * coeff_x.back()(0) * u * u + 24 * coeff_x.back()(1) * u + 6 * coeff_x.back()(2);
      j.y() = 60 * coeff_y.back()(0) * u * u + 24 * coeff_y.back()(1) * u + 6 * coeff_y.back()(2);
      j.z() = 60 * coeff_z.back()(0) * u * u + 24 * coeff_z.back()(1) * u + 6 * coeff_z.back()(2);
      return j;
    }

    if (t < times.front()) {
      j.x() = 6 * coeff_x.front()(2);
      j.y() = 6 * coeff_y.front()(2);
      j.z() = 6 * coeff_z.front()(2);
      return j;
    }

    for (size_t i = 0; i + 1 < times.size(); ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        j.x() = 60 * coeff_x[i](0) * u * u + 24 * coeff_x[i](1) * u + 6 * coeff_x[i](2);
        j.y() = 60 * coeff_y[i](0) * u * u + 24 * coeff_y[i](1) * u + 6 * coeff_y[i](2);
        j.z() = 60 * coeff_z[i](0) * u * u + 24 * coeff_z[i](1) * u + 6 * coeff_z[i](2);
        break;
      }
    }
    return j;
  }

  /// Print internal data
  void print() const {
    std::cout << "PieceWiseQuinticPol: " << times.size() - 1 << " segments\n";
    for (size_t i = 0; i + 1 < times.size(); ++i) {
      std::cout << " [" << times[i] << "," << times[i + 1] << "): "
                << "cx=" << coeff_x[i].transpose() << "  cy=" << coeff_y[i].transpose()
                << "  cz=" << coeff_z[i].transpose() << "\n";
    }
  }
};

struct state {
  // time stamp
  double t = 0.0;

  // pos, vel, accel, jerk, yaw, dyaw
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d jerk = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double dyaw = 0.0;

  // flag for tracking
  bool use_tracking_yaw = false;

  void setTimeStamp(const double data) { t = data; }

  void setPos(const double x, const double y, const double z) { pos << x, y, z; }
  void setVel(const double x, const double y, const double z) { vel << x, y, z; }
  void setAccel(const double x, const double y, const double z) { accel << x, y, z; }

  void setJerk(const double x, const double y, const double z) { jerk << x, y, z; }

  void setPos(const Eigen::Vector3d& data) { pos << data.x(), data.y(), data.z(); }

  void setVel(const Eigen::Vector3d& data) { vel << data.x(), data.y(), data.z(); }

  void setAccel(const Eigen::Vector3d& data) { accel << data.x(), data.y(), data.z(); }

  void setJerk(const Eigen::Vector3d& data) { jerk << data.x(), data.y(), data.z(); }

  void setState(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel,
                const Eigen::Vector3d& accel, const Eigen::Vector3d& jerk) {
    this->pos = pos;
    this->vel = vel;
    this->accel = accel;
    this->jerk = jerk;
  }

  void setYaw(const double data) { yaw = data; }

  void setDYaw(const double data) { dyaw = data; }

  void setZero() {
    pos = Eigen::Vector3d::Zero();
    vel = Eigen::Vector3d::Zero();
    accel = Eigen::Vector3d::Zero();
    jerk = Eigen::Vector3d::Zero();
    yaw = 0;
    dyaw = 0;
  }

  void printPos() { std::cout << "Pos= " << pos.transpose() << std::endl; }

  void print() {
    std::cout << "Time= " << t << std::endl;
    std::cout << "Pos= " << pos.transpose() << std::endl;
    std::cout << "Vel= " << vel.transpose() << std::endl;
    std::cout << "Accel= " << accel.transpose() << std::endl;
  }

  void printHorizontal() {
    std::cout << "Pos, Vel, Accel, Jerk= " << pos.transpose() << " " << vel.transpose() << " "
              << accel.transpose() << " " << jerk.transpose() << std::endl;
  }
};
