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
  PostprocessCuda(const int num_class, const float score_thresh,
                  const float nms_overlap_thresh, const int nms_pre_maxsize,
                  const int nms_post_maxsize, const int out_size_factor,
                  const int output_h, const int output_w,
                  const float pillar_x_size, const float pillar_y_size,
                  const int min_x_range, const int min_y_range, 
                  const std::map<std::string, int> &kHeadDict, 
                  const std::vector<int> &kClassNumInTask);
  ~PostprocessCuda();

  void DoPostprocessCuda(float* box_preds, float* scores, float* dir_scores,
                         std::vector<Box>& out_detections);

 private:
  const int kNumClass_;
  const float kScoreThresh_;
  const float kNmsOverlapThresh_;
  const int kNmsPreMaxsize_;
  const int kNmsPostMaxsize_;
  const int kHeadXSize_;
  const int kHeadYSize_;
  const int kOutSizeFactor_;
  const float kPillarXSize_;
  const float kPillarYSize_;
  const float kMinXRange_;
  const float kMinYRange_;

  std::map<std::string, int> kHeadDict_;
  std::vector<int> kClassNumInTask_;

  std::unique_ptr<Iou3dNmsCuda> nms_cuda_ptr_;

  int *dev_box_counter_;
  int* dev_score_idx_;
  float* sigmoid_score_;
  int* label_;
  long* host_keep_data_;
  float* host_boxes_;
  int* host_label_;
  int* host_score_idx_;
};