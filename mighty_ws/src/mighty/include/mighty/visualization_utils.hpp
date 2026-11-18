#pragma once

#include <cstddef>
#include <filesystem>
#include <utility>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

#include "mighty/mighty_type.hpp"
#include "mighty/frontend/types.hpp"

std_msgs::ColorRGBA getColorJet(double value, double minimum,
                                     double maximum);
geometry_msgs::Vector3 eigen2rosvector(const Eigen::Vector3d& vector);
geometry_msgs::Point eigen2point(const Eigen::Vector3d& vector);

visualization_msgs::MarkerArray stateVector2ColoredMarkerArray(
    const std::vector<state>& data, int type, double maximum,
    const ros::Time& stamp, const std::string& frame_id = "map");

visualization_msgs::MarkerArray stateVector2ColoredLineStripMarkerArray(
    const std::vector<state>& data, int id, const std::string& marker_namespace,
    double maximum, const ros::Time& stamp, double line_width,
    std::size_t max_points);

// Shared real-body wireframes used by the planner and baseline playback.
const std_msgs::ColorRGBA shapeWireframeLiveBodyColor();
const std_msgs::ColorRGBA shapeWireframeGhostColor();
bool loadBodyWireframe(const std::filesystem::path& path,
    std::vector<Eigen::Vector3d>& vertices,
    std::vector<std::pair<std::size_t, std::size_t>>& edges);
void appendWireframeMarkers(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::string& ns, int id_base, const geometry_msgs::Pose& pose,
    const std_msgs::ColorRGBA& color, double line_width,
    const ros::Time& stamp, const std::string& frame_id, bool frame_locked,
    visualization_msgs::MarkerArray* out);

namespace mighty {
std::vector<frontend::PoseState> sampleBodyGhostPoses(
    const std::vector<frontend::PoseState>& source, double spacing,
    std::size_t max_ghosts);
}  // namespace mighty
