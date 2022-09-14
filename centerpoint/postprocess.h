/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
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
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "iou3d_nms.h"

static const int kBoxBlockSize = 7;

struct Box {
  float x;
  float y;
  float z;
  float l;
  float w;
  float h;
  float r;
  float vx = 0.0f;  // optional
  float vy = 0.0f;  // optional
  float score;
  int label;
  bool is_drop;  // for nms
};

class PostprocessCuda {
 public:
  PostprocessCuda(const int num_class, 
                  const float score_thresh,
                  const float nms_overlap_thresh, 
                  const int nms_pre_maxsize,
                  const int nms_post_maxsize, 
                  const int downsample_size,
                  const int output_h, 
                  const int output_w,
                  const float pillar_x_size, 
                  const float pillar_y_size,
                  const int min_x_range, 
                  const int min_y_range, 
                  const std::map<std::string, int> &head_map, 
                  const std::vector<int> &num_classes_in_task);
  ~PostprocessCuda();

  void DoPostprocessCuda(float* box_preds, 
                         float* scores, 
                         float* dir_scores,
                         std::vector<Box>& out_detections);

 private:
  int num_classes_;
  float score_thresh_;
  float nms_overlap_thresh_;
  int nms_pre_max_size_;
  int nms_post_max_size_;
  int head_x_size_;
  int head_y_size_;
  int downsample_size_;
  float pillar_x_size_;
  float pillar_y_size_;
  float min_x_range_;
  float min_y_range_;
  int map_size_;
  
  std::map<std::string, int> head_map_;
  std::vector<int> num_classes_in_task_;

  std::unique_ptr<Iou3dNmsCuda> iou3d_nms_cuda_;

  float* dev_res_box_ = nullptr;
  float* dev_res_conf_ = nullptr;
  int* dev_res_cls_ = nullptr;
  int* dev_res_sorted_indices_ = nullptr;
  int* dev_res_box_num_ = nullptr;
  float* host_res_box_ = nullptr;
  float* host_res_conf_ = nullptr;
  int* host_res_cls_ = nullptr;
  int* host_res_sorted_indices_ = nullptr;
  long* host_keep_data_ = nullptr;
};