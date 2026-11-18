#include "mighty/visualization_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include <visualization_msgs/Marker.h>

std_msgs::ColorRGBA getColorJet(double value, double minimum,
                                     double maximum) {
  std_msgs::ColorRGBA color;
  color.a = 1.0;
  value = std::clamp(value, minimum, maximum);
  const double range = maximum - minimum;
  if (range <= 1.0e-9) {
    color.r = 0.8;
    color.g = 0.2;
    color.b = 0.2;
    return color;
  }

  const double normalized = (value - minimum) / range;
  constexpr double gray = 0.9;
  constexpr double saturation = 0.8;
  constexpr double brightness = 0.95;
  color.r = brightness *
            (saturation * (1.0 - normalized) + (1.0 - saturation) * gray);
  color.g = brightness *
            (saturation * normalized + (1.0 - saturation) * gray);
  color.b = brightness * (1.0 - saturation) * gray;
  return color;
}

geometry_msgs::Vector3 eigen2rosvector(const Eigen::Vector3d& vector) {
  geometry_msgs::Vector3 result;
  result.x = vector.x();
  result.y = vector.y();
  result.z = vector.z();
  return result;
}

geometry_msgs::Point eigen2point(const Eigen::Vector3d& vector) {
  geometry_msgs::Point result;
  result.x = vector.x();
  result.y = vector.y();
  result.z = vector.z();
  return result;
}

visualization_msgs::MarkerArray stateVector2ColoredMarkerArray(
    const std::vector<state>& data, int type, double maximum,
    const ros::Time&, const std::string& frame_id) {
  visualization_msgs::MarkerArray marker_array;
  if (data.empty()) return marker_array;

  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time(0);
  marker.ns = "state_vec";
  marker.id = type;
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.15;

  constexpr std::size_t kMaximumSamples = 100;
  const std::size_t step = std::max<std::size_t>(1, data.size() / kMaximumSamples);
  marker.points.reserve(2 * data.size() / step);
  marker.colors.reserve(2 * data.size() / step);

  geometry_msgs::Point previous = eigen2point(data.front().pos);
  for (std::size_t index = 0; index < data.size(); index += step) {
    const auto point = eigen2point(data[index].pos);
    const auto color = getColorJet(data[index].vel.norm(), 0.0, maximum);
    marker.points.push_back(previous);
    marker.colors.push_back(color);
    marker.points.push_back(point);
    marker.colors.push_back(color);
    previous = point;
  }

  marker_array.markers.push_back(std::move(marker));
  return marker_array;
}

visualization_msgs::MarkerArray stateVector2ColoredLineStripMarkerArray(
    const std::vector<state>& data, int id, const std::string& marker_namespace,
    double maximum, const ros::Time& stamp, double line_width,
    std::size_t max_points) {
  visualization_msgs::MarkerArray marker_array;
  if (data.size() < 2) return marker_array;

  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = stamp;
  marker.ns = marker_namespace;
  marker.id = id;
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = line_width;

  const std::size_t step =
      std::max<std::size_t>(1, data.size() / std::max<std::size_t>(2, max_points));
  marker.points.reserve((data.size() + step - 1) / step);
  marker.colors.reserve((data.size() + step - 1) / step);
  for (std::size_t index = 0; index < data.size(); index += step) {
    marker.points.push_back(eigen2point(data[index].pos));
    marker.colors.push_back(getColorJet(data[index].vel.norm(), 0.0, maximum));
  }
  if ((data.size() - 1) % step != 0) {
    marker.points.push_back(eigen2point(data.back().pos));
    marker.colors.push_back(getColorJet(data.back().vel.norm(), 0.0, maximum));
  }

  marker_array.markers.push_back(std::move(marker));
  return marker_array;
}

// Wireframe colors for shape visualization (vertices + lines). Live body is
// opaque and slightly thicker; path ghosts are transparent and thinner so
// stacked samples do not obscure the measured pose.
const std_msgs::ColorRGBA shapeWireframeLiveBodyColor() {
  std_msgs::ColorRGBA color;
  color.r = 0.05F;
  color.g = 0.75F;
  color.b = 1.0F;
  color.a = 1.0F;
  return color;
}

const std_msgs::ColorRGBA shapeWireframeGhostColor() {
  std_msgs::ColorRGBA color;
  color.r = 0.04F;
  color.g = 0.60F;
  color.b = 0.80F;
  color.a = 0.28F;
  return color;
}

// Parse the ASCII body PLY written by the PolyMap exporter (element vertex /
// property float x y z / element face / property list uchar int
// vertex_indices) into vertices and deduplicated mesh edges.
bool loadBodyWireframe(const std::filesystem::path& path,
                       std::vector<Eigen::Vector3d>& vertices,
                       std::vector<std::pair<std::size_t, std::size_t>>& edges) {
  std::ifstream input(path);
  if (!input) return false;
  std::string line;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  bool in_header = true;
  while (in_header && std::getline(input, line)) {
    if (line == "end_header") {
      in_header = false;
      break;
    }
    std::istringstream header_line(line);
    std::string keyword;
    header_line >> keyword;
    if (keyword == "element") {
      std::string kind;
      std::size_t count = 0;
      header_line >> kind >> count;
      if (kind == "vertex") vertex_count = count;
      else if (kind == "face") face_count = count;
    }
  }
  if (in_header || vertex_count == 0) return false;

  vertices.clear();
  vertices.reserve(vertex_count);
  for (std::size_t index = 0; index < vertex_count; ++index) {
    if (!std::getline(input, line)) return false;
    std::istringstream vertex_line(line);
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(vertex_line >> x >> y >> z)) return false;
    vertices.emplace_back(x, y, z);
  }

  std::set<std::pair<std::size_t, std::size_t>> unique_edges;
  for (std::size_t face = 0; face < face_count; ++face) {
    if (!std::getline(input, line)) return false;
    std::istringstream face_line(line);
    int face_size = 0;
    if (!(face_line >> face_size)) return false;
    std::vector<std::size_t> indices;
    indices.reserve(static_cast<std::size_t>(face_size));
    for (int corner = 0; corner < face_size; ++corner) {
      std::size_t index = 0;
      if (!(face_line >> index)) return false;
      indices.push_back(index);
    }
    for (std::size_t corner = 0; corner < indices.size(); ++corner) {
      const std::size_t first = indices[corner];
      const std::size_t second = indices[(corner + 1) % indices.size()];
      unique_edges.emplace(std::min(first, second), std::max(first, second));
    }
  }
  edges.assign(unique_edges.begin(), unique_edges.end());
  return true;
}

// Append the POINTS and LINE_LIST markers that render the body wireframe at
// the given pose. Both markers share id_base and id_base + 1, one color, and
// one line width. Per-vertex colors are required so RViz applies alpha on
// LINE_LIST (marker.color.a alone is ignored for lines).
//
// frame_locked must be true whenever frame_id is a moving TF frame: RViz only
// re-resolves the marker->fixed-frame transform every render frame for
// frame-locked markers. A non-frame-locked marker is transformed once, at
// reception, and then stays frozen wherever that one lookup put it.
void appendWireframeMarkers(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::string& ns, int id_base, const geometry_msgs::Pose& pose,
    const std_msgs::ColorRGBA& color, double line_width,
    const ros::Time& stamp, const std::string& frame_id, bool frame_locked,
    visualization_msgs::MarkerArray* out) {
  visualization_msgs::Marker points;
  points.header.frame_id = frame_id;
  points.header.stamp = stamp;
  points.ns = ns;
  points.id = id_base;
  points.type = visualization_msgs::Marker::POINTS;
  points.action = visualization_msgs::Marker::ADD;
  points.pose = pose;
  points.scale.x = line_width;
  points.scale.y = line_width;
  points.color = color;
  points.frame_locked = frame_locked;
  points.lifetime = ros::Duration(0, 0);
  for (const auto& vertex : vertices) {
    geometry_msgs::Point point;
    point.x = vertex.x();
    point.y = vertex.y();
    point.z = vertex.z();
    points.points.push_back(point);
    points.colors.push_back(color);
  }
  out->markers.push_back(points);

  visualization_msgs::Marker lines;
  lines.header.frame_id = frame_id;
  lines.header.stamp = stamp;
  lines.ns = ns;
  lines.id = id_base + 1;
  lines.type = visualization_msgs::Marker::LINE_LIST;
  lines.action = visualization_msgs::Marker::ADD;
  lines.pose = pose;
  lines.scale.x = line_width;
  lines.color = color;
  lines.frame_locked = frame_locked;
  lines.lifetime = ros::Duration(0, 0);
  for (const auto& [first, second] : edges) {
    if (first >= vertices.size() || second >= vertices.size()) continue;
    geometry_msgs::Point a;
    a.x = vertices[first].x();
    a.y = vertices[first].y();
    a.z = vertices[first].z();
    geometry_msgs::Point b;
    b.x = vertices[second].x();
    b.y = vertices[second].y();
    b.z = vertices[second].z();
    lines.points.push_back(a);
    lines.points.push_back(b);
    lines.colors.push_back(color);
    lines.colors.push_back(color);
  }
  out->markers.push_back(lines);
}

namespace mighty {
// Sample body poses along a polyline. Skip the prefix that coincides with the
// live start so current_body is not drawn twice at t = 0.
std::vector<frontend::PoseState> sampleBodyGhostPoses(
    const std::vector<frontend::PoseState>& source, double spacing,
    std::size_t max_ghosts) {
  std::vector<frontend::PoseState> candidates;
  if (source.empty()) return candidates;
  candidates.push_back(source.front());
  for (std::size_t index = 1; index < source.size(); ++index) {
    const auto& from = source[index - 1];
    const auto& to = source[index];
    const Eigen::Vector3d delta = to.position - from.position;
    const double distance = delta.norm();
    const double yaw_delta = std::atan2(std::sin(to.yaw - from.yaw),
                                        std::cos(to.yaw - from.yaw));
    if (distance > spacing) {
      const std::size_t interior_count =
          static_cast<std::size_t>(std::floor(distance / spacing));
      for (std::size_t sample = 1; sample <= interior_count; ++sample) {
        const double alpha =
            std::min(1.0, sample * spacing / distance);
        if (alpha >= 1.0) break;
        candidates.push_back(
            {from.position + alpha * delta, from.yaw + alpha * yaw_delta});
      }
    }
    candidates.push_back(to);
  }
  if (candidates.size() > 1) {
    const Eigen::Vector3d origin = candidates.front().position;
    auto first_keep = candidates.begin();
    while (first_keep != candidates.end() &&
           (first_keep->position - origin).norm() < 0.35) {
      ++first_keep;
    }
    candidates.erase(candidates.begin(), first_keep);
  }
  if (candidates.size() <= max_ghosts) return candidates;
  std::vector<frontend::PoseState> selected;
  selected.reserve(max_ghosts);
  for (std::size_t output = 0; output < max_ghosts; ++output) {
    selected.push_back(candidates[output * (candidates.size() - 1) /
                                  std::max<std::size_t>(1, max_ghosts - 1)]);
  }
  return selected;
}

}  // namespace mighty
