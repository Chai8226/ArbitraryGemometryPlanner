cmake_minimum_required(VERSION 3.8)
project(mighty)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Default build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED COMPONENTS roscpp rospy roslib geometry_msgs
  visualization_msgs dynus_interfaces sensor_msgs std_msgs nav_msgs
  pcl_conversions tf2 tf2_ros tf2_geometry_msgs)
find_package(Eigen3 REQUIRED)
find_package(fcl 0.7 REQUIRED CONFIG)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation surface)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(OpenMP REQUIRED)
catkin_package(INCLUDE_DIRS include LIBRARIES mighty_shape_aware_astar mighty_runtime
  CATKIN_DEPENDS roscpp roslib geometry_msgs visualization_msgs dynus_interfaces
  sensor_msgs std_msgs nav_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs)
include_directories(${catkin_INCLUDE_DIRS})
# Private FCL must remain discoverable in both devel and install spaces.
get_target_property(MIGHTY_FCL_LOCATION fcl IMPORTED_LOCATION_RELEASE)
if(NOT MIGHTY_FCL_LOCATION)
  get_target_property(MIGHTY_FCL_LOCATION fcl IMPORTED_LOCATION_NOCONFIG)
endif()
get_filename_component(MIGHTY_FCL_LIBDIR "${MIGHTY_FCL_LOCATION}" DIRECTORY)
set(CMAKE_INSTALL_RPATH "${MIGHTY_FCL_LIBDIR};${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(mighty_shape_aware_astar STATIC
  src/shape_aware_astar/planner.cpp
  src/frontend/collision_scene.cpp)
target_compile_features(mighty_shape_aware_astar PUBLIC cxx_std_17)
target_include_directories(mighty_shape_aware_astar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mighty_shape_aware_astar PUBLIC
  Eigen3::Eigen fcl nlohmann_json::nlohmann_json)
set_target_properties(mighty_shape_aware_astar PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  EXCLUDE_FROM_ALL FALSE)

add_library(mighty_runtime
  src/mighty/mighty_node.cpp
  src/mighty/mighty.cpp
  src/mighty/visualization_utils.cpp
  src/mighty/lbfgs_solver.cpp
  src/mighty/lbfgs_solver_utils.cpp
  src/mighty/whole_body_safety.cpp
  src/mighty/topp_seed.cpp)
target_compile_features(mighty_runtime PUBLIC cxx_std_17)
target_link_libraries(mighty_runtime
  mighty_shape_aware_astar
  ${PCL_LIBRARIES}
  nlohmann_json::nlohmann_json
  OpenMP::OpenMP_CXX)
target_link_libraries(mighty_runtime ${catkin_LIBRARIES})
add_dependencies(mighty_runtime ${catkin_EXPORTED_TARGETS})

add_executable(mighty src/mighty/mighty_main.cpp)
target_link_libraries(mighty mighty_runtime)
add_dependencies(mighty ${catkin_EXPORTED_TARGETS})

add_executable(fake_sim src/mighty/fake_sim.cpp)
target_compile_features(fake_sim PRIVATE cxx_std_17)
target_link_libraries(fake_sim Eigen3::Eigen)
target_link_libraries(fake_sim ${catkin_LIBRARIES})
add_dependencies(fake_sim ${catkin_EXPORTED_TARGETS})

install(TARGETS mighty fake_sim
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS mighty_shape_aware_astar mighty_runtime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES launch/known_map.launch DESTINATION share/${PROJECT_NAME}/launch)
install(FILES
  config/mighty.yaml
  config/known_map.hole.rviz
  config/known_map.forest.rviz
  config/known_map.narrow.rviz
  config/known_map.block.rviz
  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY config/params DESTINATION share/${PROJECT_NAME}/config)
install(FILES rviz/mighty.rviz DESTINATION share/${PROJECT_NAME}/rviz)

install(DIRECTORY maps shape DESTINATION share/${PROJECT_NAME})
install(DIRECTORY meshes/quadrotor DESTINATION share/${PROJECT_NAME}/meshes)
catkin_install_python(PROGRAMS scripts/run_known_map.py scripts/sim_dashboard.py
  scripts/send_goal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
