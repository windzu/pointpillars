/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include "cuda_runtime.h"
// pointpillars headers
#include "params.h"
#include "pointpillar.h"
// local
#include "det3d/det3d.hpp"

#define checkCudaErrors(status)                                       \
  {                                                                   \
    if (status != 0) {                                                \
      std::cout << "Cuda failure: " << cudaGetErrorString(status)     \
                << " at line " << __LINE__ << " in file " << __FILE__ \
                << " error status: " << status << std::endl;          \
      abort();                                                        \
    }                                                                 \
  }

void Det3D::start() {
  // pointpillars init
  get_info();

  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));
  checkCudaErrors(cudaStreamCreate(&stream_));

  nms_pred_.reserve(100);

  pointpillar_ptr_.reset(new PointPillar(model_path_, stream_));
  // PointPillar pointpillar(model_path_, stream_);

  // ros init
  pub_ = nh_.advertise<autoware_msgs::DetectedObjectArray>(result_topic_, 1);
  sub_ = nh_.subscribe(topic_, 1, &Det3D::callback, this);

  ros::spin();
}

void Det3D::callback(const sensor_msgs::PointCloud2ConstPtr &msg) {
  // load points cloud from ros msg
  frame_id_ = msg->header.frame_id;
  unsigned int length = 0;
  void *data = NULL;
  std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
  load_data(msg, &data, &length);
  buffer.reset((char *)data);

  // format data to 4 channels
  float *points = (float *)buffer.get();
  size_t points_size = length / sizeof(float) / 4;

  ROS_DEBUG_STREAM("find points num: " << points_size);

  float *points_data = nullptr;
  unsigned int points_data_size = points_size * 4 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
  checkCudaErrors(
      cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
  checkCudaErrors(cudaDeviceSynchronize());

  cudaEventRecord(start_, stream_);

  pointpillar_ptr_->doinfer(points_data, points_size, nms_pred_);
  cudaEventRecord(stop_, stream_);
  cudaEventSynchronize(stop_);
  cudaEventElapsedTime(&elapsed_time_, start_, stop_);
  ROS_DEBUG_STREAM("TIME: pointpillar: " << elapsed_time_ << " ms.");
  checkCudaErrors(cudaFree(points_data));

  ROS_DEBUG_STREAM("Bndbox objs: " << nms_pred_.size());
  // std::string save_file_name = Save_Dir + index_str + ".txt";
  pub_box_pred(nms_pred_);

  nms_pred_.clear();

  ROS_DEBUG_STREAM(">>>>>>>>>>>");
}

void Det3D::get_info(void) {
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int Det3D::load_data(const sensor_msgs::PointCloud2ConstPtr &msg, void **data,
                     unsigned int *length) {
  // convert sensor_msgs::PointCloud to array
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*msg, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromPCLPointCloud2(pcl_pc2, *pcl_pc);

  // use the first 4 channels
  int num_points = pcl_pc->points.size();
  int num_channels = 4;
  *length = num_points * num_channels * sizeof(float);
  *data = malloc(*length);
  float *data_ptr = (float *)(*data);
  for (int i = 0; i < num_points; i++) {
    data_ptr[i * num_channels + 0] = pcl_pc->points[i].x;
    data_ptr[i * num_channels + 1] = pcl_pc->points[i].y;
    data_ptr[i * num_channels + 2] = pcl_pc->points[i].z;
    data_ptr[i * num_channels + 3] = pcl_pc->points[i].intensity / 255.0;
  }
  return 0;
}

void Det3D::pub_box_pred(std::vector<Bndbox> boxes) {
  autoware_msgs::DetectedObjectArray objects;
  objects.header.frame_id = frame_id_;
  objects.header.stamp = ros::Time::now();
  objects.header.seq = 0;
  for (const auto box : boxes) {
    if (box.score < score_threshold_) continue;

    autoware_msgs::DetectedObject obj;
    obj.header = objects.header;
    obj.label = box.id;  // class id
    // obj.label = "car";  // debug

    obj.score = box.score;
    obj.pose.position.x = box.x;
    obj.pose.position.y = box.y;
    obj.pose.position.z = box.z;

    // NOTE: box 的 l w h 对应的是车的 宽 长 高
    // 我们 dimensions 的 x y z 对应的是车的 长 宽 高
    // 所以这里要交换一下
    obj.dimensions.x = box.w;
    obj.dimensions.y = box.l;
    obj.dimensions.z = box.h;

    // 将yaw角转换为四元数
    float yaw = box.rt;
    tf2::Quaternion quat_tf;
    quat_tf.setRPY(0, 0, yaw);
    obj.pose.orientation = tf2::toMsg(quat_tf);
    objects.objects.push_back(obj);
  }
  pub_.publish(objects);
  return;
}
