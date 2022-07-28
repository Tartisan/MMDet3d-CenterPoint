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

#include <memory>
#include <vector>

#include "nms.h"

class PostprocessCuda {
 public:
  PostprocessCuda(const int num_threads, const float float_min,
                  const float float_max, const int num_class,
                  const int num_anchor_per_cls, const float score_threshold,
                  const float nms_overlap_threshold, const int nms_pre_maxsize,
                  const int nms_post_maxsize, const int num_box_corners,
                  const int num_input_box_feature,
                  const int num_output_box_feature);
  ~PostprocessCuda() {}

  void DoPostprocessCuda(const float* box_preds, const float* box_scores,
                         float* host_box, float* host_score,
                         int* host_filtered_count,
                         std::vector<float>& out_detection,
                         std::vector<int>& out_label,
                         std::vector<float>& out_score);

 private:
  const int num_threads_;
  const float float_min_;
  const float float_max_;
  const int num_class_;
  const int num_per_cls_;
  const float score_threshold_;
  const float nms_overlap_threshold_;
  const int nms_pre_maxsize_;
  const int nms_post_maxsize_;
  const int num_box_corners_;
  const int num_input_box_feature_;
  const int num_output_box_feature_;
  const std::vector<std::vector<int>> multihead_label_mapping_;

  std::unique_ptr<NmsCuda> nms_cuda_ptr_;
};