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

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
 * @author Yan haixu
 * Contact: just github.com/hova88
 * @date 2021/04/30
 */

/**
 * @author Ye xiubo
 * Contact:github.com/speshowBUAA
 * @date 2022/01/05
 */

// headers in CUDA
#include <thrust/sort.h>

// headers in local files
#include <stdio.h>

#include "common.h"
#include "postprocess.h"

template <typename T>
void swap_warp(T& a, T& b, T& swp) {
  swp = a;
  a = b;
  b = swp;
}
void quicksort_warp(float* score, int* index, int start, int end) {
  if (start >= end) return;
  float pivot = score[end];
  float value_swp;
  int index_swp;
  // set a pointer to divide array into two parts
  // one part is smaller than pivot and another larger
  int pointer = start;
  for (int i = start; i < end; i++) {
    if (score[i] > pivot) {
      if (pointer != i) {
        // swap score[i] with score[pointer]
        // score[pointer] behind larger than pivot
        swap_warp<float>(score[i], score[pointer], value_swp);
        swap_warp<int>(index[i], index[pointer], index_swp);
      }
      pointer++;
    }
  }
  // swap back pivot to proper position
  swap_warp<float>(score[end], score[pointer], value_swp);
  swap_warp<int>(index[end], index[pointer], index_swp);
  quicksort_warp(score, index, start, pointer - 1);
  quicksort_warp(score, index, pointer + 1, end);
  return;
}

void quicksort_kernel(float* score, int* indexes, int len) {
  quicksort_warp(score, indexes, 0, len - 1);
}

PostprocessCuda::PostprocessCuda(
    const int num_threads, const float float_min, const float float_max,
    const int num_class, const int num_per_cls, const float score_threshold,
    const float nms_overlap_threshold, const int nms_pre_maxsize,
    const int nms_post_maxsize, const int num_box_corners,
    const int num_input_box_feature,
    const int num_output_box_feature)
    : num_threads_(num_threads),                         // 64
      float_min_(float_min),                             //
      float_max_(float_max),                             //
      num_class_(num_class),                             // 3
      num_per_cls_(num_per_cls),                         // 4096
      score_threshold_(score_threshold),                 // 0.1
      nms_overlap_threshold_(nms_overlap_threshold),     // 0.25
      nms_pre_maxsize_(nms_pre_maxsize),                 // 4096
      nms_post_maxsize_(nms_post_maxsize),               // 500
      num_box_corners_(num_box_corners),                 // 8
      num_input_box_feature_(num_input_box_feature),     // 7
      num_output_box_feature_(num_output_box_feature) {  // 7
  nms_cuda_ptr_.reset(
      new NmsCuda(num_threads_, num_box_corners_, nms_overlap_threshold_));
}

void PostprocessCuda::DoPostprocessCuda(const float* box_preds,
                                        const float* box_scores,
                                        float* host_box, float* host_score,
                                        int* host_filtered_count,
                                        std::vector<float>& out_detection,
                                        std::vector<int>& out_label,
                                        std::vector<float>& out_score) {
  GPU_CHECK(cudaMemcpy(host_box, box_preds,
                       num_output_box_feature_ * num_per_cls_ * sizeof(float),
                       cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(host_score, box_scores,
                       num_class_ * num_per_cls_ * sizeof(float),
                       cudaMemcpyDeviceToHost));

  // hardcode for class_map as {0, 12 , 34 , 5 ,67 ,89}
  for (int class_idx = 0; class_idx < num_class_; ++class_idx) {
    // init parameter
    host_filtered_count[class_idx] = 0;

    float host_filtered_score[nms_pre_maxsize_];    // 3840
    float host_filtered_box[nms_pre_maxsize_ * 7];  // 3840 * 7
    for (size_t idx = 0; idx < num_per_cls_; idx++) {
      float score = host_score[idx * num_class_ + class_idx];
      // filter out boxes which threshold less than score_threshold
      if (score > score_threshold_ &&
          host_filtered_count[class_idx] < nms_pre_maxsize_) {
        host_filtered_score[host_filtered_count[class_idx]] = score;
        // dim_idx = {x,y,z,dx,dy,dz,yaw}
        for (size_t dim_idx = 0; dim_idx < 7; dim_idx++) {
          host_filtered_box[host_filtered_count[class_idx] * 7 + dim_idx] =
              host_box[idx * num_output_box_feature_ + dim_idx];
        }
        host_filtered_count[class_idx] += 1;
      }
    }
    // printf("host_filter_count[%d] = %d\n", 
    //        class_idx, host_filtered_count[class_idx]);
    if (host_filtered_count[class_idx] <= 0) continue;

    float host_sorted_filtered_box[host_filtered_count[class_idx] * 7];
    float host_sorted_filtered_score[host_filtered_count[class_idx]];

    for (int ith_box = 0; ith_box < host_filtered_count[class_idx]; ++ith_box) {
      host_sorted_filtered_score[ith_box] = host_filtered_score[ith_box];
      host_sorted_filtered_box[ith_box * 7 + 0] =
          host_filtered_box[ith_box * 7 + 0];
      host_sorted_filtered_box[ith_box * 7 + 1] =
          host_filtered_box[ith_box * 7 + 1];
      host_sorted_filtered_box[ith_box * 7 + 2] =
          host_filtered_box[ith_box * 7 + 2];
      host_sorted_filtered_box[ith_box * 7 + 3] =
          host_filtered_box[ith_box * 7 + 3];
      host_sorted_filtered_box[ith_box * 7 + 4] =
          host_filtered_box[ith_box * 7 + 4];
      host_sorted_filtered_box[ith_box * 7 + 5] =
          host_filtered_box[ith_box * 7 + 5];
      host_sorted_filtered_box[ith_box * 7 + 6] =
          host_filtered_box[ith_box * 7 + 6];
    }

    // host to device for nms cuda
    // In fact, this cuda calc is also not necessary.
    // After each category is filtered by sigmoid, there are only about 100, up
    // to 1000 boxes left behind. Use CUDA_NMS will never faster than CPU_NMS
    // TODO : use cpu_nms replace cuda_nms
    float* dev_sorted_filtered_box;
    float* dev_sorted_filtered_score;

    GPU_CHECK(cudaMalloc((void**)&dev_sorted_filtered_box,
                         host_filtered_count[class_idx] * 7 * sizeof(float)));
    GPU_CHECK(cudaMalloc((void**)&dev_sorted_filtered_score,
                         host_filtered_count[class_idx] * sizeof(float)));

    GPU_CHECK(cudaMemcpy(dev_sorted_filtered_box, host_sorted_filtered_box,
                         host_filtered_count[class_idx] * 7 * sizeof(float),
                         cudaMemcpyHostToDevice));

    GPU_CHECK(cudaMemcpy(dev_sorted_filtered_score, host_sorted_filtered_score,
                         host_filtered_count[class_idx] * sizeof(float),
                         cudaMemcpyHostToDevice));

    int num_box_for_nms = min(nms_pre_maxsize_, host_filtered_count[class_idx]);
    long keep_inds[num_box_for_nms];  // index of kept box
    memset(keep_inds, 0, num_box_for_nms * sizeof(int));

    int det_num_boxes_per_class = 0;
    nms_cuda_ptr_->DoNmsCuda(num_box_for_nms, dev_sorted_filtered_box,
                             keep_inds, &det_num_boxes_per_class);
    det_num_boxes_per_class = min(det_num_boxes_per_class, nms_post_maxsize_);

    // recopy to host
    cudaMemcpy(host_sorted_filtered_box, dev_sorted_filtered_box,
               host_filtered_count[class_idx] * 7 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sorted_filtered_score, dev_sorted_filtered_score,
               host_filtered_count[class_idx] * sizeof(float),
               cudaMemcpyDeviceToHost);

    // int det_num_filtered_boxes_pre_class = 0;
    for (int box_idx = 0; box_idx < det_num_boxes_per_class; ++box_idx) {
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 0]);
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 1]);
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 2]);
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 3]);
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 4]);
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 5]);
      out_detection.emplace_back(
          host_sorted_filtered_box[keep_inds[box_idx] * 7 + 6]);
      out_score.emplace_back(host_sorted_filtered_score[keep_inds[box_idx]]);
      out_label.emplace_back(class_idx);
    }

    GPU_CHECK(cudaFree(dev_sorted_filtered_box));
    GPU_CHECK(cudaFree(dev_sorted_filtered_score));
  }
}
