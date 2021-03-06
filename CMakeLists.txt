cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (3D_object_tracking
        src/main.cpp
        src/fusion/camFusion_Student.cpp
        src/FinalProject_Camera.cpp
        src/lidar/lidarData.cpp
        src/camera/matching2D_Student.cpp
        src/camera/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})

add_executable (Output_3D_object_tracking
        src/main_csv.cpp
        src/fusion/camFusion_Student.cpp
        src/FinalProject_Camera.cpp
        src/lidar/lidarData.cpp
        src/camera/matching2D_Student.cpp
        src/camera/objectDetection2D.cpp)
target_link_libraries (Output_3D_object_tracking ${OpenCV_LIBRARIES})