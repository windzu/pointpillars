cmake_minimum_required(VERSION 3.5)

project(det3d)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
# opencv
find_package(OpenCV REQUIRED)
# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# cuda and tensorrt
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

# ros
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  sensor_msgs
  tf2
  autoware_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} 
  CATKIN_DEPENDS roscpp roslib std_msgs sensor_msgs tf2 autoware_msgs
  DEPENDS OpenCV PCL
)

include_directories( ${catkin_INCLUDE_DIRS} )

# pointpillars依赖tensorrt和cuda，无法加入catkin_package中，需要单独编译
include_directories(./pointpillars/include)
add_subdirectory(./pointpillars)


include_directories(./include)
aux_source_directory(./src SRC)
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME}
  pointpillars nvinfer cudart  
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES}
)
add_dependencies(${PROJECT_NAME}  
  ${catkin_EXPORTED_TARGETS}
)

install(TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

