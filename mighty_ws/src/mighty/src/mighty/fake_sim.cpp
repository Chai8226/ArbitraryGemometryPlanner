/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <mighty/runtime_guards.hpp>
#include <mighty/ros_parameters.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>

#include <Eigen/Core>

#include "dynus_interfaces/Goal.h"
#include "dynus_interfaces/State.h"
#include "geometry_msgs/TransformStamped.h"
#include <ros/ros.h>
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/Marker.h"

class FakeSim final {
 public:
  FakeSim() : private_nh_("~") {
    const auto start = mighty::readParameter<std::vector<double>>(private_nh_, "start_pos", {0.0, 0.0, 1.0});
    if (start.size() != 3) {
      throw std::invalid_argument("start_pos must contain exactly three values");
    }
    map_frame_id_ = mighty::readParameter<std::string>(private_nh_, "map_frame_id", "map");
    publish_marker_ = mighty::readParameter<int>(private_nh_, "visual_level", 1) > 0;

    state_.header.frame_id = map_frame_id_;
    state_.pos.x = start[0];
    state_.pos.y = start[1];
    state_.pos.z = start[2];
    tf2::Quaternion orientation;
    orientation.setRPY(0.0, 0.0, mighty::readParameter<double>(private_nh_, "start_yaw", 0.0));
    state_.quat.x = orientation.x();
    state_.quat.y = orientation.y();
    state_.quat.z = orientation.z();
    state_.quat.w = orientation.w();

    namespace_ = nh_.getNamespace();
    if (!namespace_.empty() && namespace_.front() == '/') {
      namespace_.erase(namespace_.begin());
    }
    base_frame_id_ = namespace_ + "/base_link";

    state_publisher_ = nh_.advertise<dynus_interfaces::State>("state", 1);
    marker_publisher_ = nh_.advertise<visualization_msgs::Marker>("drone_marker", 1);
    goal_subscription_ = nh_.subscribe("goal", 10, &FakeSim::goalCallback, this);
    timer_ = nh_.createWallTimer(ros::WallDuration(0.01), [this](const ros::WallTimerEvent&) { publish(); });

    ROS_INFO( "Known-map kinematic simulator initialized at %.3f %.3f %.3f",
                state_.pos.x, state_.pos.y, state_.pos.z);


  }

 private:
  void goalCallback(const dynus_interfaces::Goal::ConstPtr goal) {
    state_.pos = goal->p;
    state_.vel = goal->v;

    Eigen::Vector3d thrust(goal->a.x, goal->a.y, goal->a.z + 9.81);
    const double norm = thrust.norm();
    if (norm <= 1e-9) {
      return;
    }
    thrust /= norm;
    const double denominator = std::sqrt(2.0 * (1.0 + thrust.z()));
    if (denominator <= 1e-9) {
      return;
    }
    tf2::Quaternion tilt(-thrust.y() / denominator,
                         thrust.x() / denominator,
                         0.0,
                         (1.0 + thrust.z()) / denominator);
    tf2::Quaternion yaw(0.0, 0.0, std::sin(goal->yaw / 2.0),
                        std::cos(goal->yaw / 2.0));
    const tf2::Quaternion orientation = tilt * yaw;
    state_.quat.x = orientation.x();
    state_.quat.y = orientation.y();
    state_.quat.z = orientation.z();
    state_.quat.w = orientation.w();
  }

  void publish() {
    const auto stamp = ros::Time::now();
    state_.header.stamp = stamp;
    state_publisher_.publish(state_);

    geometry_msgs::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = map_frame_id_;
    transform.child_frame_id = base_frame_id_;
    transform.transform.translation.x = state_.pos.x;
    transform.transform.translation.y = state_.pos.y;
    transform.transform.translation.z = state_.pos.z;
    transform.transform.rotation = state_.quat;
    transform_broadcaster_.sendTransform(transform);

    if (publish_marker_) {
      marker_publisher_.publish(droneMarker(stamp));
    }
  }

  visualization_msgs::Marker droneMarker(const ros::Time& stamp) const {
    visualization_msgs::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = map_frame_id_;
    marker.ns = "known_map_uav";
    marker.id = 1;
    marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = state_.pos.x;
    marker.pose.position.y = state_.pos.y;
    marker.pose.position.z = state_.pos.z;
    marker.pose.orientation = state_.quat;
    marker.mesh_resource = "package://mighty/meshes/quadrotor/quadrotor.dae";
    marker.mesh_use_embedded_materials = true;
    marker.scale.x = 0.75;
    marker.scale.y = 0.75;
    marker.scale.z = 0.75;
    return marker;
  }

  dynus_interfaces::State state_;
  std::string namespace_;
  std::string map_frame_id_;
  std::string base_frame_id_;
  bool publish_marker_{true};
  ros::Publisher state_publisher_;
  ros::Publisher marker_publisher_;
  ros::Subscriber goal_subscription_;
  ros::WallTimer timer_;
  ros::NodeHandle nh_, private_nh_;
  tf2_ros::TransformBroadcaster transform_broadcaster_;
};

int main(int argc, char** argv) {
  // Exit if the launch parent disappears.
  if (mighty::armParentDeathSignal()) return 0;

  ros::init(argc, argv, "fake_sim");
  try {
    FakeSim node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("%s", error.what());
    return 1;
  }
  return 0;
}
