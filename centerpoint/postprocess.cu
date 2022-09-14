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

__device__ float sigmoid_gpu(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void DecodeObjectKernel(const int map_size,
                                   const float score_thresh,
                                   const int nms_pre_max_size,
                                   const float min_x_range,
                                   const float min_y_range,
                                   const float pillar_x_size,
                                   const float pillar_y_size,
                                   const int head_x_size,
                                   const int head_y_size,
                                   const int downsample_size,
                                   const int num_class_in_task,
                                   const int cls_range,
                                   const float* reg,
                                   const float* hei,
                                   const float* dim,
                                   const float* rot,
                                   const float* cls,
                                   float* res_box,
                                   float* res_conf,
                                   int* res_cls,
                                   int* res_box_num) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= map_size) return;

  float max_score = cls[idx];
  int label = cls_range;
  for (int i = 1; i < num_class_in_task; ++i) {
    float cur_score = cls[idx + i * map_size];
    if (cur_score > max_score) {
      max_score = cur_score;
      label = i + cls_range;
    }
  }

  int coor_x = idx % head_x_size;
  int coor_y = idx / head_x_size;

  float conf = sigmoid_gpu(max_score);
  if (conf > score_thresh) {
    int cur_valid_box_id = atomicAdd(res_box_num, 1);
    if (cur_valid_box_id < nms_pre_max_size) {
      res_box[cur_valid_box_id * kBoxBlockSize + 0] = 
          (reg[idx + 0 * map_size] + coor_x) * downsample_size * pillar_x_size + min_x_range;
      res_box[cur_valid_box_id * kBoxBlockSize + 1] = 
          (reg[idx + 1 * map_size] + coor_y) * downsample_size * pillar_y_size + min_y_range;
      res_box[cur_valid_box_id * kBoxBlockSize + 2] = hei[idx];
      res_box[cur_valid_box_id * kBoxBlockSize + 3] = expf(dim[idx + 0 * map_size]);
      res_box[cur_valid_box_id * kBoxBlockSize + 4] = expf(dim[idx + 1 * map_size]);
      res_box[cur_valid_box_id * kBoxBlockSize + 5] = expf(dim[idx + 2 * map_size]);
      res_box[cur_valid_box_id * kBoxBlockSize + 6] = atan2f(rot[idx], rot[idx + map_size]);
      res_conf[cur_valid_box_id] = conf;
      res_cls[cur_valid_box_id] = label;
    }
  }
}

PostprocessCuda::PostprocessCuda(const int num_class, const float score_thresh,
                                 const float nms_overlap_thresh,
                                 const int nms_pre_maxsize,
                                 const int nms_post_maxsize,
                                 const int downsample_size, const int output_h,
                                 const int output_w, const float pillar_x_size,
                                 const float pillar_y_size,
                                 const int min_x_range, const int min_y_range,
                                 const std::map<std::string, int>& head_dict,
                                 const std::vector<int>& class_num_in_task)
    : num_classes_(num_class),
      score_thresh_(score_thresh),
      nms_overlap_thresh_(nms_overlap_thresh),
      nms_pre_max_size_(nms_pre_maxsize),
      nms_post_max_size_(nms_post_maxsize),
      downsample_size_(downsample_size),
      head_x_size_(output_h),
      head_y_size_(output_w),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range) {
  head_map_ = head_dict;
  num_classes_in_task_ = class_num_in_task;
  map_size_ = head_x_size_ * head_y_size_;

  iou3d_nms_cuda_.reset(new Iou3dNmsCuda(head_x_size_, head_y_size_, nms_overlap_thresh_));

  GPU_CHECK(cudaMalloc((void**)&dev_res_box_, sizeof(float) * nms_pre_max_size_ * kBoxBlockSize));
  GPU_CHECK(cudaMalloc((void**)&dev_res_conf_, sizeof(float) * nms_pre_max_size_));
  GPU_CHECK(cudaMalloc((void**)&dev_res_cls_, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMalloc((void**)&dev_res_sorted_indices_, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMalloc((void**)&dev_res_box_num_, sizeof(int)));
  GPU_CHECK(cudaMallocHost((void**)&host_res_box_, sizeof(float) * nms_pre_max_size_ * kBoxBlockSize));
  GPU_CHECK(cudaMallocHost((void**)&host_res_conf_, sizeof(float) * nms_pre_max_size_));
  GPU_CHECK(cudaMallocHost((void**)&host_res_cls_, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMallocHost((void**)&host_res_sorted_indices_, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMallocHost((void**)&host_keep_data_, sizeof(long) * nms_pre_max_size_));
  
  GPU_CHECK(cudaMemset(dev_res_box_, 0.f, sizeof(float) * nms_pre_max_size_ * kBoxBlockSize));
  GPU_CHECK(cudaMemset(dev_res_conf_, 0.f, sizeof(float) * nms_pre_max_size_));
  GPU_CHECK(cudaMemset(dev_res_cls_, 0, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMemset(dev_res_sorted_indices_, 0, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMemset(dev_res_box_num_, 0, sizeof(int)));
  GPU_CHECK(cudaMemset(host_res_box_, 0.f, sizeof(float) * nms_pre_max_size_ * kBoxBlockSize));
  GPU_CHECK(cudaMemset(host_res_conf_, 0.f, sizeof(float) * nms_pre_max_size_));
  GPU_CHECK(cudaMemset(host_res_cls_, 0, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMemset(host_res_sorted_indices_, 0, sizeof(int) * nms_pre_max_size_));
  GPU_CHECK(cudaMemset(host_keep_data_, 0L, sizeof(long) * nms_pre_max_size_));
}

void PostprocessCuda::DoPostprocessCuda(float* bbox_preds, float* cls_scores,
                                        float* dir_scores,
                                        std::vector<Box>& out_detections) {
  int box_range = head_map_["reg"] + head_map_["height"] + head_map_["dim"];
  int dir_range = head_map_["rot"];
  std::vector<int> cls_range{0};

  for (size_t i = 0; i < num_classes_in_task_.size(); i++) {
    float* reg = bbox_preds + (box_range * i + 0) * map_size_;
    float* hei = bbox_preds + (box_range * i + head_map_["reg"]) * map_size_;
    float* dim = bbox_preds + (box_range * i + head_map_["reg"] + head_map_["height"]) * map_size_;
    float* rot = dir_scores + (dir_range * i) * map_size_;
    float* cls = cls_scores + cls_range[i] * map_size_;
    cls_range.push_back(cls_range[i] + num_classes_in_task_[i]);

    GPU_CHECK(cudaMemset(dev_res_box_, 0.f, sizeof(float) * nms_pre_max_size_ * kBoxBlockSize));
    GPU_CHECK(cudaMemset(dev_res_conf_, 0.f, sizeof(float) * nms_pre_max_size_));
    GPU_CHECK(cudaMemset(dev_res_cls_, 0, sizeof(int) * nms_pre_max_size_));
    GPU_CHECK(cudaMemset(dev_res_box_num_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(host_keep_data_, -1, sizeof(long) * nms_pre_max_size_));

    dim3 block(NUM_THREADS);
    dim3 grid(DIVUP(map_size_, NUM_THREADS));
    DecodeObjectKernel<<<grid, block>>>(map_size_,
                                        score_thresh_,
                                        nms_pre_max_size_,
                                        min_x_range_,
                                        min_y_range_,
                                        pillar_x_size_,
                                        pillar_y_size_,
                                        head_x_size_,
                                        head_y_size_,
                                        downsample_size_,
                                        num_classes_in_task_[i],
                                        cls_range[i],
                                        reg,
                                        hei,
                                        dim,
                                        rot,
                                        cls,
                                        dev_res_box_,
                                        dev_res_conf_,
                                        dev_res_cls_,
                                        dev_res_box_num_);
    
    int box_num_pre = 0;
    GPU_CHECK(cudaMemcpy(&box_num_pre, dev_res_box_num_, sizeof(int), cudaMemcpyDeviceToHost));

    thrust::sequence(thrust::device, dev_res_sorted_indices_, dev_res_sorted_indices_ + box_num_pre);
    thrust::sort_by_key(thrust::device,
                        dev_res_conf_,
                        dev_res_conf_ + box_num_pre,
                        dev_res_sorted_indices_,
                        thrust::greater<float>());

    int box_num_post = iou3d_nms_cuda_->DoIou3dNms(box_num_pre,
                                                   dev_res_box_, 
                                                   dev_res_sorted_indices_,
                                                   host_keep_data_);
    
    box_num_post = box_num_post > nms_post_max_size_ ? nms_post_max_size_ : box_num_post;
    std::cout << "task " << i << " gets " << box_num_pre 
              << " objects before nms, and " << box_num_post
              << " after nms" << std::endl;
    
    GPU_CHECK(cudaMemcpy(host_res_box_, dev_res_box_, sizeof(float) * box_num_pre * kBoxBlockSize, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(host_res_conf_, dev_res_conf_, sizeof(float) * box_num_pre, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(host_res_cls_, dev_res_cls_, sizeof(int) * box_num_pre, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(host_res_sorted_indices_, dev_res_sorted_indices_, sizeof(int) * box_num_pre, cudaMemcpyDeviceToHost));

    for (auto j = 0; j < box_num_post; j++) {
      int k = host_keep_data_[j];
      int idx = host_res_sorted_indices_[k];

      Box box;
      box.x = host_res_box_[idx * kBoxBlockSize + 0];
      box.y = host_res_box_[idx * kBoxBlockSize + 1];
      box.z = host_res_box_[idx * kBoxBlockSize + 2];
      box.l = host_res_box_[idx * kBoxBlockSize + 3];
      box.w = host_res_box_[idx * kBoxBlockSize + 4];
      box.h = host_res_box_[idx * kBoxBlockSize + 5];
      box.r = host_res_box_[idx * kBoxBlockSize + 6];
      box.label = host_res_cls_[idx];
      box.score = host_res_conf_[k];
      box.z -= box.h * 0.5; // bottom height
      out_detections.push_back(box);
    }
  }
}

PostprocessCuda::~PostprocessCuda() {
  GPU_CHECK(cudaFree(dev_res_box_));
  GPU_CHECK(cudaFree(dev_res_conf_));
  GPU_CHECK(cudaFree(dev_res_cls_));
  GPU_CHECK(cudaFree(dev_res_sorted_indices_));
  GPU_CHECK(cudaFree(dev_res_box_num_));
  GPU_CHECK(cudaFreeHost(host_res_box_));
  GPU_CHECK(cudaFreeHost(host_res_conf_));
  GPU_CHECK(cudaFreeHost(host_res_cls_));
  GPU_CHECK(cudaFreeHost(host_res_sorted_indices_));
  GPU_CHECK(cudaFreeHost(host_keep_data_));
}
