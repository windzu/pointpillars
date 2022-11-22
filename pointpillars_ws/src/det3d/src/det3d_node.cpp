#include "det3d/det3d.hpp"
#include "ros/ros.h"
int main(int argc, char** argv) {
  ros::init(argc, argv, "det3d_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  Det3D det3d(nh, pnh);
  det3d.start();

  return 0;
}
