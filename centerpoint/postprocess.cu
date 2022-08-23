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

#include <stdio.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "common.h"
#include "postprocess.h"

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void process_kernel(float* dim, float* score, float* sigmoid_score,
                               int* label, const int num_class,
                               const int score_range, const int length,
                               const float score_thresh, int* box_counter) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= length) return;

  float max_score = score[idx];
  int label_id = 0 + score_range;
  for (int i = 1; i < num_class; ++i) {
    float cur_score = score[idx + i * length];
    if (cur_score > max_score) {
      max_score = cur_score;
      label_id = i + score_range;
    }
  }
  sigmoid_score[idx] = sigmoid(max_score);
  label[idx] = label_id;
  if (sigmoid_score[idx] > score_thresh) {
    atomicAdd(box_counter, 1);
    dim[idx + 0 * length] = expf(dim[idx + 0 * length]);
    dim[idx + 1 * length] = expf(dim[idx + 1 * length]);
    dim[idx + 2 * length] = expf(dim[idx + 2 * length]);
    // printf("idx: %d, score: %.2f, label: %d, lwh: %.2f, %.2f, %.2f\n", idx,
    //        sigmoid_score[idx], label[idx], dim[idx + 0 * length],
    //        dim[idx + 1 * length], dim[idx + 2 * length]);
  }
}

struct is_greater {
  is_greater(float thre) : _thre(thre) {}
  __host__ __device__ bool operator()(const float& x) { return x >= _thre; }
  float _thre;
};

int find_valid_score_kernel(float* score, float thre, const int output_h,
                            const int output_w) {
  // thrust::device_vector<float> score_vec(score, score + output_h * output_w);
  // return thrust::count_if(thrust::device, score_vec.begin(), score_vec.end(),
  //                         is_greater(thre));
  return thrust::count_if(thrust::device, score, score + output_h * output_w,
                          is_greater(thre));
}

void sort_by_key_kernel(float* keys, int* values, int size) {
  thrust::sequence(thrust::device, values, values + size);
  thrust::sort_by_key(thrust::device, keys, keys + size, values,
                      thrust::greater<float>());
}

void gather_kernel(float* host_boxes, int* host_label, float* reg,
                   float* height, float* dim, float* rot, float* sorted_score,
                   int* label, int* dev_indexs, long* host_keep_indexs,
                   int box_num_bef, int box_num_aft, const int length) {
  // copy keep_indexs from host to device
  thrust::device_vector<long> dev_keep_indexs(host_keep_indexs,
                                              host_keep_indexs + box_num_aft);
  // gather keeped indexs after nms
  thrust::device_vector<int> dev_indexs_bef(dev_indexs,
                                            dev_indexs + box_num_bef);
  thrust::device_vector<int> dev_indexs_aft(box_num_aft);
  thrust::gather(dev_keep_indexs.begin(), dev_keep_indexs.end(),
                 dev_indexs_bef.begin(), dev_indexs_aft.begin());
  // gather boxes, score, label
  thrust::device_vector<float> out_boxes(box_num_aft * 9);
  thrust::device_vector<int> out_label(box_num_aft);
  // gather x, y
  thrust::device_vector<float> reg_vec(reg, reg + length * 2);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(), reg_vec.begin(),
                 out_boxes.begin());
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 reg_vec.begin() + length, out_boxes.begin() + box_num_aft);
  // gather height
  thrust::device_vector<float> height_vec(height, height + length);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 height_vec.begin(), out_boxes.begin() + 2 * box_num_aft);
  // gather dim
  thrust::device_vector<float> dim_vec(dim, dim + 3 * length);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 dim_vec.begin() + length * 0,
                 out_boxes.begin() + 3 * box_num_aft);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 dim_vec.begin() + length * 1,
                 out_boxes.begin() + 4 * box_num_aft);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 dim_vec.begin() + length * 2,
                 out_boxes.begin() + 5 * box_num_aft);
  // gather rotation
  thrust::device_vector<float> rot_vec(rot, rot + 2 * length);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 rot_vec.begin() + length * 0,
                 out_boxes.begin() + 6 * box_num_aft);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 rot_vec.begin() + length * 1,
                 out_boxes.begin() + 7 * box_num_aft);
  // gather score
  thrust::device_vector<float> sorted_score_vec(sorted_score,
                                                sorted_score + 1 * length);
  thrust::gather(dev_keep_indexs.begin(), dev_keep_indexs.end(),
                 sorted_score_vec.begin() + length * 0,
                 out_boxes.begin() + 8 * box_num_aft);
  // gather label
  thrust::device_vector<int> label_vec(label, label + 1 * length);
  thrust::gather(dev_indexs_aft.begin(), dev_indexs_aft.end(),
                 label_vec.begin() + length * 0, out_label.begin());
  // copy values from device to host
  thrust::copy(out_boxes.begin(), out_boxes.end(), host_boxes);
  thrust::copy(out_label.begin(), out_label.end(), host_label);
}

PostprocessCuda::PostprocessCuda(const int num_class, const float score_thresh,
                                 const float nms_overlap_thresh,
                                 const int nms_pre_maxsize,
                                 const int nms_post_maxsize,
                                 const int out_size_factor, const int output_h,
                                 const int output_w, const float pillar_x_size,
                                 const float pillar_y_size,
                                 const int min_x_range, const int min_y_range,
                                 const std::map<std::string, int>& head_dict,
                                 const std::vector<int>& class_num_in_task)
    : kNumClass_(num_class),
      kScoreThresh_(score_thresh),
      kNmsOverlapThresh_(nms_overlap_thresh),
      kNmsPreMaxsize_(nms_pre_maxsize),
      kNmsPostMaxsize_(nms_post_maxsize),
      kOutSizeFactor_(out_size_factor),
      kHeadXSize_(output_h),
      kHeadYSize_(output_w),
      kPillarXSize_(pillar_x_size),
      kPillarYSize_(pillar_y_size),
      kMinXRange_(min_x_range),
      kMinYRange_(min_y_range) {
  kHeadDict_ = head_dict;
  kClassNumInTask_ = class_num_in_task;

  nms_cuda_ptr_.reset(new Iou3dNmsCuda(
      kHeadXSize_, kHeadYSize_, kNmsOverlapThresh_, kOutSizeFactor_,
      kPillarXSize_, kPillarYSize_, kMinXRange_, kMinYRange_));

  GPU_CHECK(cudaMalloc((void**)&dev_box_counter_, sizeof(int)));
  GPU_CHECK(cudaMemset(dev_box_counter_, 0, sizeof(int)));

  GPU_CHECK(cudaMalloc((void**)&dev_score_idx_,
                       kHeadXSize_ * kHeadYSize_ * sizeof(int)));
  GPU_CHECK(
      cudaMemset(dev_score_idx_, -1, kHeadXSize_ * kHeadYSize_ * sizeof(int)));

  GPU_CHECK(
      cudaMallocHost((void**)&host_keep_data_, kNmsPreMaxsize_ * sizeof(long)));
  GPU_CHECK(cudaMemset(host_keep_data_, -1, kNmsPreMaxsize_ * sizeof(long)));

  GPU_CHECK(cudaMallocHost((void**)&host_boxes_,
                           kNmsPostMaxsize_ * 9 * sizeof(float)));
  GPU_CHECK(cudaMemset(host_boxes_, 0, kNmsPostMaxsize_ * 9 * sizeof(float)));

  GPU_CHECK(
      cudaMallocHost((void**)&host_label_, kNmsPostMaxsize_ * sizeof(int)));
  GPU_CHECK(cudaMemset(host_label_, -1, kNmsPostMaxsize_ * sizeof(int)));

  GPU_CHECK(cudaMallocHost((void**)&host_score_idx_,
                           kHeadXSize_ * kHeadYSize_ * sizeof(int)));
  GPU_CHECK(
      cudaMemset(host_score_idx_, -1, kHeadXSize_ * kHeadYSize_ * sizeof(int)));

  GPU_CHECK(cudaMalloc((void**)&sigmoid_score_,
                       kHeadXSize_ * kHeadYSize_ * sizeof(float)));
  GPU_CHECK(
      cudaMalloc((void**)&label_, kHeadXSize_ * kHeadYSize_ * sizeof(int)));
}

void PostprocessCuda::DoPostprocessCuda(float* bbox_preds, float* scores,
                                        float* dir_scores,
                                        std::vector<Box>& out_detections) {
  int bbox_range = kHeadDict_["reg"] + kHeadDict_["height"] + kHeadDict_["dim"];
  int dir_range = kHeadDict_["rot"];
  std::vector<int> score_range{0};

  for (size_t i = 0; i < kClassNumInTask_.size(); i++) {
    float* reg = bbox_preds + (bbox_range * i + 0) * kHeadXSize_ * kHeadYSize_;
    float* height = bbox_preds + (bbox_range * i + kHeadDict_["reg"]) *
                                     kHeadXSize_ * kHeadYSize_;
    float* dim = bbox_preds +
                 (bbox_range * i + kHeadDict_["reg"] + kHeadDict_["height"]) *
                     kHeadXSize_ * kHeadYSize_;
    float* score = scores + score_range[i] * kHeadXSize_ * kHeadYSize_;
    float* rot = dir_scores + (dir_range * i) * kHeadXSize_ * kHeadYSize_;
    score_range.push_back(score_range[i] + kClassNumInTask_[i]);

    GPU_CHECK(cudaMemset(sigmoid_score_, 0,
                         kHeadXSize_ * kHeadYSize_ * sizeof(float)));
    GPU_CHECK(cudaMemset(label_, -1, kHeadXSize_ * kHeadYSize_ * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_box_counter_, 0, sizeof(int)));
    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(kHeadXSize_ * kHeadYSize_, NUM_THREADS));
    process_kernel<<<grid, block>>>(
        dim, score, sigmoid_score_, label_, kClassNumInTask_[i], score_range[i],
        kHeadXSize_ * kHeadYSize_, kScoreThresh_, dev_box_counter_);

    // int box_num_bef = find_valid_score_kernel(sigmoid_score_, kScoreThresh_,
    //                                           kHeadXSize_, kHeadYSize_);
    int box_num_bef = 0;
    GPU_CHECK(cudaMemcpy(&box_num_bef, dev_box_counter_, sizeof(int),
                         cudaMemcpyDeviceToHost));
    std::cout << "num boxes before " << box_num_bef << std::endl;
    box_num_bef = box_num_bef > kNmsPreMaxsize_ ? kNmsPreMaxsize_ : box_num_bef;

    sort_by_key_kernel(sigmoid_score_, dev_score_idx_,
                       kHeadXSize_ * kHeadYSize_);

    GPU_CHECK(cudaMemset(host_keep_data_, -1, kNmsPreMaxsize_ * sizeof(long)));
    int box_num_aft = nms_cuda_ptr_->DoNmsCuda(
        reg, height, dim, rot, dev_score_idx_, host_keep_data_, box_num_bef);
    box_num_aft =
        box_num_aft > kNmsPostMaxsize_ ? kNmsPostMaxsize_ : box_num_aft;
    std::cout << "num boxes after " << box_num_aft << std::endl;

    gather_kernel(host_boxes_, host_label_, reg, height, dim, rot,
                  sigmoid_score_, label_, dev_score_idx_, host_keep_data_,
                  box_num_bef, box_num_aft, kHeadXSize_ * kHeadYSize_);

    GPU_CHECK(cudaMemcpy(host_score_idx_, dev_score_idx_,
                         box_num_bef * sizeof(int), cudaMemcpyDeviceToHost));
    for (auto j = 0; j < box_num_aft; j++) {
      int k = host_keep_data_[j];
      // std::cout << j << ", " << k << ", \n";
      int idx = host_score_idx_[k];
      int xIdx = idx % kHeadYSize_;
      int yIdx = idx / kHeadYSize_;

      Box box;
      box.x = (host_boxes_[j + 0 * box_num_aft] + xIdx) * kOutSizeFactor_ *
                  kPillarXSize_ +
              kMinXRange_;
      box.y = (host_boxes_[j + 1 * box_num_aft] + yIdx) * kOutSizeFactor_ *
                  kPillarYSize_ +
              kMinYRange_;
      box.z = host_boxes_[j + 2 * box_num_aft];
      box.l = host_boxes_[j + 3 * box_num_aft];
      box.w = host_boxes_[j + 4 * box_num_aft];
      box.h = host_boxes_[j + 5 * box_num_aft];
      box.z -= box.h * 0.5;
      float r_sin = host_boxes_[j + 6 * box_num_aft];
      float r_cos = host_boxes_[j + 7 * box_num_aft];
      box.r = atan2(r_sin, r_cos);
      box.score = host_boxes_[j + 8 * box_num_aft];
      box.label = host_label_[j];
      out_detections.push_back(box);
    }
  }
}

PostprocessCuda::~PostprocessCuda() {
  GPU_CHECK(cudaFree(dev_box_counter_));
  GPU_CHECK(cudaFree(dev_score_idx_));
  GPU_CHECK(cudaFree(sigmoid_score_));
  GPU_CHECK(cudaFree(label_));
  GPU_CHECK(cudaFreeHost(host_keep_data_));
  GPU_CHECK(cudaFreeHost(host_boxes_));
  GPU_CHECK(cudaFreeHost(host_label_));
  GPU_CHECK(cudaFreeHost(host_score_idx_));
}
