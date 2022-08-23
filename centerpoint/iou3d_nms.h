/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2022.
*/
#pragma once

#include <iostream>

// #define DEBUG
const int THREADS_PER_BLOCK = 16;
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;

class Iou3dNmsCuda {
 public:
  Iou3dNmsCuda(const int output_h, const int output_w,
               const float nms_overlap_thresh, const float out_size_factor,
               const float pillar_x_size, const float pillar_y_size,
               const float min_x_range, const float min_y_range);
  ~Iou3dNmsCuda() = default;

  int DoNmsCuda(const float* reg, const float* height, const float* dim,
                const float* rot, const int* indexs, long* host_keep_data,
                int boxes_num);

 private:
  const int kHeadXSize_;
  const int kHeadYSize_;
  const float kNmsOverlapThresh_;
  const float kOutSizeFactor_;
  const float kPillarXSize_;
  const float kPillarYSize_;
  const float kMinXRange_;
  const float kMinYRange_;
};
