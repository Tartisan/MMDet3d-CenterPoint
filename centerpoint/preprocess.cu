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

// headers in STL
#include <stdio.h>

// headers in local files
#include "common.h"
#include "preprocess.h"

/**
 *  统计 [468x468] 中每个栅格是否有 point 占据（mask），如果占据，mask
 * 中对应栅格的点云 数量 +1, 同时将 point 的信息填充进
 * dev_pillar_point_feature_in_coors
 */
__global__ void make_pillar_histo_kernel(
    const float* dev_points, float* dev_pillar_point_feature_in_coors,
    int* mask, const int num_points, const int max_points_per_pillar,
    const int grid_x_size, const int grid_y_size, const int grid_z_size,
    const float min_x_range, const float min_y_range, const float min_z_range,
    const float max_x_range, const float max_y_range, const float max_z_range,
    const float pillar_x_size, const float pillar_y_size,
    const float pillar_z_size, const int num_point_feature) {
  int th_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (th_i >= num_points) {
    return;
  }

  float x = dev_points[th_i * num_point_feature + 0];
  float y = dev_points[th_i * num_point_feature + 1];
  float z = dev_points[th_i * num_point_feature + 2];

  if (x < min_x_range || x >= max_x_range || y < min_y_range ||
      y >= max_y_range || z < min_z_range || z >= max_z_range) {
    return;
  }

  int x_coor = floor((x - min_x_range) / pillar_x_size);
  int y_coor = floor((y - min_y_range) / pillar_y_size);

  int pillar_idx = y_coor * grid_x_size + x_coor;
  int count = atomicAdd(&mask[pillar_idx], 1);
  if (count < max_points_per_pillar) {
    int ind = pillar_idx * max_points_per_pillar * num_point_feature +
              count * num_point_feature;
    for (int i = 0; i < num_point_feature; ++i) {
      dev_pillar_point_feature_in_coors[ind + i] =
          dev_points[th_i * num_point_feature + i];
    }
  }
}

/**
 * 根据上步统计得到的 mask 和 dev_pillar_point_feature_in_coors，
 * 计算 mask 中非空栅格的数量，并保留栅格中最多 max_points_per_pillar
 * 个点云的特征
 */
__global__ void make_pillar_index_kernel(
    float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature,
    int* dev_pillar_coors, int* mask, int* dev_pillar_count,
    float* dev_num_points_per_pillar, const int max_pillars,
    const int max_points_per_pillar, const int grid_x_size,
    const int num_point_feature) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int pillar_coors_index = y * grid_x_size + x;
  int num_points = mask[pillar_coors_index];
  if (num_points == 0) {
    return;
  }

  int cur_pillar_id = atomicAdd(dev_pillar_count, 1);
  if (cur_pillar_id < max_pillars) {
    num_points = num_points >= max_points_per_pillar ? max_points_per_pillar
                                                     : num_points;
    dev_num_points_per_pillar[cur_pillar_id] = num_points;

    for (int i = 0; i < num_points; i++) {
      int in_index = pillar_coors_index * max_points_per_pillar + i;
      int out_index = cur_pillar_id * max_points_per_pillar + i;
      for (int j = 0; j < num_point_feature; ++j) {
        dev_pillar_point_feature[out_index * num_point_feature + j] =
            dev_pillar_point_feature_in_coors[in_index * num_point_feature + j];
      }
    }

    int4 idx = {0, 0, y, x};
    ((int4*)dev_pillar_coors)[cur_pillar_id] = idx;
  }
}

__global__ void pillar_mean_kernel(float* dev_points_mean,
                                   const int num_point_feature,
                                   const float* dev_pillar_point_feature,
                                   const float* dev_num_points_per_pillar,
                                   int max_pillars, int max_points_per_pillar) {
  extern __shared__ float temp[];
  int ith_pillar = blockIdx.x;
  int ith_point = threadIdx.x;
  int axis = threadIdx.y;

  int reduce_size = max_points_per_pillar > 32 ? 64 : 32;
  temp[threadIdx.x * 3 + axis] =
      dev_pillar_point_feature[ith_pillar * max_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + axis];
  if (threadIdx.x < reduce_size - max_points_per_pillar) {
    temp[(threadIdx.x + max_points_per_pillar) * 3 + axis] =
        0.0f;  //--> dummy placeholds will set as 0
  }
  __syncthreads();
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

  if (ith_point >= num_points_at_this_pillar) {
    return;
  }

  for (unsigned int d = reduce_size >> 1; d > 0.6; d >>= 1) {
    if (ith_point < d) {
      temp[ith_point * 3 + axis] += temp[(ith_point + d) * 3 + axis];
    }
    __syncthreads();
  }

  if (ith_point == 0) {
    dev_points_mean[ith_pillar * 3 + axis] =
        temp[ith_point + axis] / num_points_at_this_pillar;
  }
}

__device__ void warpReduce(volatile float* sdata, int ith_point, int axis) {
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 8) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 4) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 2) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 1) * blockDim.y + axis];
}

__global__ void make_pillar_mean_kernel(float* dev_points_mean,
                                        const int num_point_feature,
                                        const float* dev_pillar_point_feature,
                                        const float* dev_num_points_per_pillar,
                                        int max_pillars,
                                        int max_points_pre_pillar) {
  extern __shared__ float temp[];
  unsigned int ith_pillar = blockIdx.x;  // { 0 , 1, 2, ... , 10000+}
  unsigned int ith_point = threadIdx.x;  // { 0 , 1, 2, ...,9}
  unsigned int axis = threadIdx.y;
  unsigned int idx_pre =
      ith_pillar * max_points_pre_pillar * num_point_feature +
      ith_point * num_point_feature;
  unsigned int idx_post =
      ith_pillar * max_points_pre_pillar * num_point_feature +
      (ith_point + blockDim.x) * num_point_feature;

  temp[ith_point * blockDim.y + axis] = 0.0;
  unsigned int num_points_at_this_pillar =
      dev_num_points_per_pillar[ith_pillar];

  // if (ith_point < num_points_at_this_pillar / 2) {
  temp[ith_point * blockDim.y + axis] =
      dev_pillar_point_feature[idx_pre + axis] +
      dev_pillar_point_feature[idx_post + axis];
  // }
  __syncthreads();

  // do reduction in shared mem
  // Sequential addressing. This solves the bank conflicts as
  // the threads now access shared memory with a stride of one
  // 32-bit word (unsigned int) now, which does not cause bank
  // conflicts
  warpReduce(temp, ith_point, axis);

  // // write result for this block to global mem
  if (ith_point == 0)
    dev_points_mean[ith_pillar * blockDim.y + axis] =
        temp[ith_point * blockDim.y + axis] / num_points_at_this_pillar;
}

// dev_num_points_per_pillar 根据Max_points_per_pillar过滤后每个位置上的点云数目
// dev_pillar_point_feature 过滤后的点云
// dev_pillar_coors 过滤后点云对应的坐标
// dev_pfe_gather_feature_ 加入全部点云中心3维坐标特征 再加全部体素中心3维坐标
__global__ void gather_point_feature_kernel(
    const int max_num_pillars_, const int max_num_points_per_pillar,
    const int num_point_feature, const float min_x_range,
    const float min_y_range, const float min_z_range, const float pillar_x_size,
    const float pillar_y_size, const float pillar_z_size, const int grid_x_size,
    const float* dev_pillar_point_feature,
    const float* dev_num_points_per_pillar, const int* dev_pillar_coors,
    float* dev_points_mean, float* dev_pfe_gather_feature_) {
  int ith_pillar = blockIdx.x;
  int ith_point = threadIdx.x;
  // int kNumPointFeature = 5;
  // int num_gather_feature = 11;   // multihead_pp 是11
  int num_gather_feature = 10;  // mmdet3d 是10
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

  if (ith_point >= num_points_at_this_pillar) {
    return;
  }

  int x_coor =
      floor((dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                          num_point_feature +
                                      ith_point * num_point_feature + 0] -
             min_x_range) /
            pillar_x_size);
  int y_coor =
      floor((dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                          num_point_feature +
                                      ith_point * num_point_feature + 1] -
             min_y_range) /
            pillar_y_size);
  // int pillar_ind = dev_coor_to_voxelidx[y_coor * grid_x_size + x_coor];
  // dev_x_coors[pillar_ind] = x_coor;
  // dev_y_coors[pillar_ind] = y_coor;

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 0] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 0];

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 1] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 1];

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 2] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 2];

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 3] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 3];

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 3] = 0.0f;

  // f_cluster = voxel_features[:, :, :3] - points_mean
  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 4] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 0] -
      dev_points_mean[ith_pillar * 3 + 0];

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 5] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 1] -
      dev_points_mean[ith_pillar * 3 + 1];

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 6] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 2] -
      dev_points_mean[ith_pillar * 3 + 2];

  // f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].
  //    to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 7] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 0] -
      (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size +
       (pillar_x_size / 2 + min_x_range));

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 8] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 1] -
      (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size +
       (pillar_y_size / 2 + min_y_range));

  dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar *
                              num_gather_feature +
                          ith_point * num_gather_feature + 9] =
      dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar *
                                   num_point_feature +
                               ith_point * num_point_feature + 2] -
      (dev_pillar_coors[ith_pillar * 4 + 1] * pillar_z_size +
       (pillar_z_size / 2 + min_z_range));
}

PreprocessPointsCuda::PreprocessPointsCuda(
    const int num_threads, const int max_num_pillars,
    const int max_points_per_pillar, const int num_point_feature,
    const int grid_x_size, const int grid_y_size, const int grid_z_size,
    const float pillar_x_size, const float pillar_y_size,
    const float pillar_z_size, const float min_x_range, const float min_y_range,
    const float min_z_range, const float max_x_range, const float max_y_range,
    const float max_z_range)
    : num_threads_(num_threads),
      max_num_pillars_(max_num_pillars),
      max_points_per_pillar_(max_points_per_pillar),
      num_point_feature_(num_point_feature),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range),
      max_x_range_(max_x_range),
      max_y_range_(max_y_range),
      max_z_range_(max_z_range) {
  GPU_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&dev_pillar_point_feature_in_coors_),
                 grid_y_size_ * grid_x_size_ * max_points_per_pillar_ *
                     num_point_feature_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&mask_),
                       grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_), sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_mean_),
                       max_num_pillars_ * 3 * sizeof(float)));
}

PreprocessPointsCuda::~PreprocessPointsCuda() {
  GPU_CHECK(cudaFree(dev_pillar_point_feature_in_coors_));
  GPU_CHECK(cudaFree(mask_));
  GPU_CHECK(cudaFree(dev_pillar_count_));
  GPU_CHECK(cudaFree(dev_points_mean_));
}

void PreprocessPointsCuda::DoPreprocessPointsCuda(
    const float* dev_points, const int in_num_points,
    float* dev_num_points_per_pillar, float* dev_pillar_point_feature,
    int* dev_pillar_coors, int* host_pillar_count,
    float* dev_pfe_gather_feature) {
  // initialize paraments
  GPU_CHECK(cudaMemset(dev_pillar_point_feature_in_coors_, 0,
                       grid_y_size_ * grid_x_size_ * max_points_per_pillar_ *
                           num_point_feature_ * sizeof(float)));
  GPU_CHECK(cudaMemset(mask_, 0, grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_pillar_count_, 0, sizeof(int)));
  GPU_CHECK(
      cudaMemset(dev_points_mean_, 0, max_num_pillars_ * 3 * sizeof(float)));

  // dev_pillar_point_feature_in_coors_ 将点云按照大小顺序排好，
  // pillar_count_histo 储存每一个位置上的点云数目
  int num_block = DIVUP(in_num_points, num_threads_);
  make_pillar_histo_kernel<<<num_block, num_threads_>>>(
      dev_points, dev_pillar_point_feature_in_coors_, mask_, in_num_points,
      max_points_per_pillar_, grid_x_size_, grid_y_size_, grid_z_size_,
      min_x_range_, min_y_range_, min_z_range_, max_x_range_, max_y_range_,
      max_z_range_, pillar_x_size_, pillar_y_size_, pillar_z_size_,
      num_point_feature_);

  make_pillar_index_kernel<<<grid_x_size_, grid_y_size_>>>(
      dev_pillar_point_feature_in_coors_, dev_pillar_point_feature,
      dev_pillar_coors, mask_, dev_pillar_count_, dev_num_points_per_pillar,
      max_num_pillars_, max_points_per_pillar_, grid_x_size_,
      num_point_feature_);
  // dev_pillar_count_ pillar总数
  // mask_ 储存每一个位置上的点云数目
  // dev_num_points_per_pillar
  // 根据Max_points_per_pillar过滤后每个位置上的点云数目 dev_x_coors
  // 每个序号对应点云的x grid坐标 dev_y_coors 每个序号对应点云的y grid坐标
  // dev_pillar_count_ 过滤后有效的pillar数量

  GPU_CHECK(cudaMemcpy(host_pillar_count, dev_pillar_count_, 1 * sizeof(int),
                       cudaMemcpyDeviceToHost));
  std::cout << "find pillar num: " << host_pillar_count[0] << std::endl;

  dim3 mean_block(max_points_per_pillar_, 3);  //(32,3)
  pillar_mean_kernel<<<host_pillar_count[0], mean_block,
                       64 * 3 * sizeof(float)>>>(
      dev_points_mean_, num_point_feature_, dev_pillar_point_feature,
      dev_num_points_per_pillar, max_num_pillars_, max_points_per_pillar_);

  gather_point_feature_kernel<<<max_num_pillars_, max_points_per_pillar_>>>(
      max_num_pillars_, max_points_per_pillar_, num_point_feature_,
      min_x_range_, min_y_range_, min_z_range_, pillar_x_size_, pillar_y_size_,
      pillar_z_size_, grid_x_size_, dev_pillar_point_feature,
      dev_num_points_per_pillar, dev_pillar_coors, dev_points_mean_,
      dev_pfe_gather_feature);
}
