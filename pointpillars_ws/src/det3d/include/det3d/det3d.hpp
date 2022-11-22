/*
 * @Author: windzu
 * @Date: 2022-04-01 09:53:39
 * @LastEditTime: 2022-05-24 10:13:31
 * @LastEditors: windzu
 * @Description:
 * @FilePath: /tld/include/tld/tld.hpp
 * @Copyright (C) 2021-2022 plusgo Company Limited. All rights reserved.
 * @Licensed under the Apache License, Version 2.0 (the License)
 */
#pragma once
#include <math.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// cuda
#include "cuda_runtime.h"
// ros
#include "autoware_msgs/DetectedObjectArray.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
// pcl
#include "pcl/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
// pointpillars headers
#include "params.h"
#include "pointpillar.h"

#define __APP_NAME__ "Det3D"

class Det3D {
 public:
  Det3D(ros::NodeHandle nh, ros::NodeHandle pnh) : nh_(nh), pnh_(pnh) {
    pnh_.param("topic", topic_, std::string("/lidar"));
    pnh_.param("result_topic", result_topic_, std::string("/lidar/result"));
    pnh_.param("model_path", model_path_, std::string(""));
    pnh_.param("score_threshold", score_threshold_, static_cast<float>(0.3));
  }
  void start();

 private:
  // callback
  void callback(const sensor_msgs::PointCloud2ConstPtr &msg);

  void get_info(void);
  int load_data(const sensor_msgs::PointCloud2ConstPtr &msg, void **data,
                unsigned int *length);
  void pub_box_pred(std::vector<Bndbox> boxes);

 private:
  // ros
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // roslaunch params
  std::string topic_;
  std::string result_topic_;
  std::string model_path_;
  float score_threshold_;

  // cuda
  cudaEvent_t start_, stop_;
  cudaStream_t stream_ = NULL;

  // pointpillars
  std::vector<Bndbox> nms_pred_;
  Params params_;
  float elapsed_time_ = 0.0f;
  std::unique_ptr<PointPillar> pointpillar_ptr_;

  // msgs subscriber
  ros::Subscriber sub_;
  ros::Publisher pub_;

  // temp
  std::string frame_id_ = "map";
};
