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
 *  统计 [468x468] 中每个栅格是否有 point 占据（pillar_count_histo），如果占据，pillar_count_histo
 * 中对应栅格的点云 数量 +1, 同时将 point 的信息填充进
 * dev_pillar_point_feature_in_coors
 */
__global__ void PillarHistoKernel(const int num_points, 
                                  const int max_points_in_voxel,
                                  const int grid_x_size, 
                                  const int grid_y_size, 
                                  const int grid_z_size,
                                  const float min_x_range, 
                                  const float min_y_range, 
                                  const float min_z_range,
                                  const float max_x_range, 
                                  const float max_y_range, 
                                  const float max_z_range,
                                  const float pillar_x_size, 
                                  const float pillar_y_size,
                                  const float pillar_z_size, 
                                  const int point_feature_dim,
                                  const float* dev_points, 
                                  float* dev_pillar_point_feature_in_coors,
                                  int* pillar_count_histo) {
  int th_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (th_i >= num_points) {
    return;
  }

  float x = dev_points[th_i * point_feature_dim + 0];
  float y = dev_points[th_i * point_feature_dim + 1];
  float z = dev_points[th_i * point_feature_dim + 2];

  if (x < min_x_range || x >= max_x_range || y < min_y_range ||
      y >= max_y_range || z < min_z_range || z >= max_z_range) {
    return;
  }

  int x_coor = floor((x - min_x_range) / pillar_x_size);
  int y_coor = floor((y - min_y_range) / pillar_y_size);

  int pillar_idx = y_coor * grid_x_size + x_coor;
  int count = atomicAdd(&pillar_count_histo[pillar_idx], 1);
  if (count < max_points_in_voxel) {
    int ind = pillar_idx * max_points_in_voxel * point_feature_dim + count * point_feature_dim;
    for (int i = 0; i < point_feature_dim; ++i) {
      dev_pillar_point_feature_in_coors[ind + i] = dev_points[th_i * point_feature_dim + i];
    }
  }
}

/**
 * 根据上步统计得到的 pillar_count_histo 和 dev_pillar_point_feature_in_coors，
 * 计算 pillar_count_histo 中非空栅格的数量，并保留栅格中最多 max_points_in_voxel
 * 个点云的特征
 */
__global__ void PillarBaseFeatureKernel(const int max_pillars,
                                        const int max_points_in_voxel, 
                                        const int grid_x_size,
                                        const int point_feature_dim,
                                        const float* dev_pillar_point_feature_in_coors, 
                                        const int* pillar_count_histo, 
                                        float* dev_pillar_point_feature,
                                        int* dev_pillar_coors, 
                                        int* dev_pillar_num,
                                        int* dev_num_points_in_voxel) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int pillar_coors_index = y * grid_x_size + x;
  int num_points = pillar_count_histo[pillar_coors_index];
  if (num_points == 0) {
    return;
  }

  int cur_pillar_id = atomicAdd(dev_pillar_num, 1);
  if (cur_pillar_id < max_pillars) {
    num_points = num_points >= max_points_in_voxel ? max_points_in_voxel
                                                     : num_points;
    dev_num_points_in_voxel[cur_pillar_id] = num_points;

    for (int i = 0; i < num_points; i++) {
      int in_index = pillar_coors_index * max_points_in_voxel + i;
      int out_index = cur_pillar_id * max_points_in_voxel + i;
      for (int j = 0; j < point_feature_dim; ++j) {
        dev_pillar_point_feature[out_index * point_feature_dim + j] =
            dev_pillar_point_feature_in_coors[in_index * point_feature_dim + j];
      }
    }

    int4 idx = {0, 0, y, x};
    ((int4*)dev_pillar_coors)[cur_pillar_id] = idx;
  }
}

__global__ void PillarMeanFeatureKernel(const int max_pillars, 
                                        const int max_points_in_voxel, 
                                        const int point_feature_dim,
                                        const int* dev_num_points_in_voxel,
                                        const float* dev_pillar_point_feature,
                                        float* dev_points_mean) {
  extern __shared__ float temp[];
  int ith_pillar = blockIdx.x;
  int ith_point = threadIdx.x;
  int axis = threadIdx.y;

  int reduce_size = max_points_in_voxel > 32 ? 64 : 32;
  temp[threadIdx.x * 3 + axis] =
      dev_pillar_point_feature[ith_pillar * max_points_in_voxel * point_feature_dim + ith_point * point_feature_dim + axis];
  if (threadIdx.x < reduce_size - max_points_in_voxel) {
    temp[(threadIdx.x + max_points_in_voxel) * 3 + axis] =
        0.0f;  //--> dummy placeholds will set as 0
  }
  __syncthreads();
  int num_points_at_this_pillar = dev_num_points_in_voxel[ith_pillar];

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
    dev_points_mean[ith_pillar * 3 + axis] = temp[ith_point + axis] / num_points_at_this_pillar;
  }
}

__device__ void warpReduce(volatile float* sdata, int ith_point, int axis) {
  sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 8) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 4) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 2) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 1) * blockDim.y + axis];
}

__global__ void make_pillar_mean_kernel(float* dev_points_mean,
                                        const int point_feature_dim,
                                        const float* dev_pillar_point_feature,
                                        const int* dev_num_points_in_voxel,
                                        int max_pillars,
                                        int max_points_pre_pillar) {
  extern __shared__ float temp[];
  unsigned int ith_pillar = blockIdx.x;  // { 0 , 1, 2, ... , 10000+}
  unsigned int ith_point = threadIdx.x;  // { 0 , 1, 2, ...,9}
  unsigned int axis = threadIdx.y;
  unsigned int idx_pre = ith_pillar * max_points_pre_pillar * point_feature_dim + ith_point * point_feature_dim;
  unsigned int idx_post = ith_pillar * max_points_pre_pillar * point_feature_dim + (ith_point + blockDim.x) * point_feature_dim;

  temp[ith_point * blockDim.y + axis] = 0.0;
  unsigned int num_points_at_this_pillar = dev_num_points_in_voxel[ith_pillar];

  // if (ith_point < num_points_at_this_pillar / 2) {
  temp[ith_point * blockDim.y + axis] = dev_pillar_point_feature[idx_pre + axis] + dev_pillar_point_feature[idx_post + axis];
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
    dev_points_mean[ith_pillar * blockDim.y + axis] = temp[ith_point * blockDim.y + axis] / num_points_at_this_pillar;
}

// dev_num_points_in_voxel 根据Max_points_per_pillar过滤后每个位置上的点云数目
// dev_pillar_point_feature 过滤后的点云
// dev_pillar_coors 过滤后点云对应的坐标
// dev_voxel_feature 加入全部点云中心3维坐标特征 再加全部体素中心3维坐标
__global__ void GatherFeatureKernel(const int max_voxel_num, 
                                    const int max_points_in_voxel,
                                    const int point_feature_dim, 
                                    const float min_x_range,
                                    const float min_y_range, 
                                    const float min_z_range, 
                                    const float pillar_x_size,
                                    const float pillar_y_size, 
                                    const float pillar_z_size, 
                                    const int grid_x_size,
                                    const float* dev_pillar_point_feature,
                                    const int* dev_num_points_in_voxel, 
                                    const int* dev_pillar_coors,
                                    const float* dev_points_mean, 
                                    float* dev_voxel_feature) {
  int ith_pillar = blockIdx.x;
  int ith_point = threadIdx.x;

  int num_gather_feature = 10;  // mmdet3d 是10
  int num_points_at_this_pillar = dev_num_points_in_voxel[ith_pillar];

  if (ith_point >= num_points_at_this_pillar) {
    return;
  }

  int ith_base_feature = ith_pillar * max_points_in_voxel * point_feature_dim + ith_point * point_feature_dim;
  int ith_voxel_feature = ith_pillar * max_points_in_voxel * num_gather_feature + ith_point * num_gather_feature;
  float x = dev_pillar_point_feature[ith_base_feature + 0];
  float y = dev_pillar_point_feature[ith_base_feature + 1];
  float z = dev_pillar_point_feature[ith_base_feature + 2];
  float intensity = dev_pillar_point_feature[ith_base_feature + 3];
  float x_offset = pillar_x_size / 2 + min_x_range;
  float y_offset = pillar_y_size / 2 + min_y_range;
  float z_offset = pillar_z_size / 2 + min_z_range;

  dev_voxel_feature[ith_voxel_feature + 0] = x;
  dev_voxel_feature[ith_voxel_feature + 1] = y;
  dev_voxel_feature[ith_voxel_feature + 2] = z;
  dev_voxel_feature[ith_voxel_feature + 3] = intensity;
  // f_cluster = voxel_features[:, :, :3] - points_mean
  dev_voxel_feature[ith_voxel_feature + 4] = x - dev_points_mean[ith_pillar * 3 + 0];
  dev_voxel_feature[ith_voxel_feature + 5] = y - dev_points_mean[ith_pillar * 3 + 1];
  dev_voxel_feature[ith_voxel_feature + 6] = z - dev_points_mean[ith_pillar * 3 + 2];
  // f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].
  //    to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
  dev_voxel_feature[ith_voxel_feature + 7] = x - (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size + x_offset);
  dev_voxel_feature[ith_voxel_feature + 8] = y - (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size + y_offset);
  dev_voxel_feature[ith_voxel_feature + 9] = z - (dev_pillar_coors[ith_pillar * 4 + 1] * pillar_z_size + z_offset);
}

PreprocessPointsCuda::PreprocessPointsCuda(const int max_voxel_num,
                                           const int max_points_in_voxel, 
                                           const int point_feature_dim,
                                           const int grid_x_size, 
                                           const int grid_y_size, 
                                           const int grid_z_size,
                                           const float pillar_x_size, 
                                           const float pillar_y_size,
                                           const float pillar_z_size, 
                                           const float min_x_range, 
                                           const float min_y_range,
                                           const float min_z_range, 
                                           const float max_x_range, 
                                           const float max_y_range,
                                           const float max_z_range)
    : max_voxel_num_(max_voxel_num),
      max_points_in_voxel_(max_points_in_voxel),
      point_feature_dim_(point_feature_dim),
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
  map_size_ = grid_x_size_ * grid_y_size_;
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_point_feature_in_coors_), 
                       map_size_ * max_points_in_voxel_ * point_feature_dim_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&pillar_count_histo_), map_size_ * sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_num_), sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_mean_), max_voxel_num_ * 3 * sizeof(float)));
}

PreprocessPointsCuda::~PreprocessPointsCuda() {
  GPU_CHECK(cudaFree(dev_pillar_point_feature_in_coors_));
  GPU_CHECK(cudaFree(pillar_count_histo_));
  GPU_CHECK(cudaFree(dev_pillar_num_));
  GPU_CHECK(cudaFree(dev_points_mean_));
}

void PreprocessPointsCuda::DoPreprocessPointsCuda(const float* dev_points, 
                                                  const int in_num_points,
                                                  int* dev_num_points_in_voxel, 
                                                  float* dev_pillar_point_feature,
                                                  int* dev_pillar_coors, 
                                                  int* host_pillar_count,
                                                  float* dev_voxel_feature) {
  // initialize paraments
  GPU_CHECK(cudaMemset(dev_pillar_point_feature_in_coors_, 0,
                       map_size_ * max_points_in_voxel_ * point_feature_dim_ * sizeof(float)));
  GPU_CHECK(cudaMemset(pillar_count_histo_, 0, map_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_pillar_num_, 0, sizeof(int)));
  GPU_CHECK(cudaMemset(dev_points_mean_, 0, max_voxel_num_ * 3 * sizeof(float)));

  // dev_pillar_point_feature_in_coors_ 将点云按照大小顺序排好，
  // pillar_count_histo 储存每一个位置上的点云数目
  int num_block = DIVUP(in_num_points, kNumThreads);
  PillarHistoKernel<<<num_block, kNumThreads>>>(in_num_points,
                                                max_points_in_voxel_, 
                                                grid_x_size_, 
                                                grid_y_size_, 
                                                grid_z_size_,
                                                min_x_range_, 
                                                min_y_range_, 
                                                min_z_range_, 
                                                max_x_range_, 
                                                max_y_range_,
                                                max_z_range_, 
                                                pillar_x_size_, 
                                                pillar_y_size_, 
                                                pillar_z_size_,
                                                point_feature_dim_,
                                                dev_points, 
                                                dev_pillar_point_feature_in_coors_, 
                                                pillar_count_histo_);

  PillarBaseFeatureKernel<<<grid_x_size_, grid_y_size_>>>(max_voxel_num_, 
                                                          max_points_in_voxel_, 
                                                          grid_x_size_,
                                                          point_feature_dim_, 
                                                          dev_pillar_point_feature_in_coors_, 
                                                          pillar_count_histo_, 
                                                          dev_pillar_point_feature,
                                                          dev_pillar_coors, 
                                                          dev_pillar_num_, 
                                                          dev_num_points_in_voxel);

  GPU_CHECK(cudaMemcpy(host_pillar_count, dev_pillar_num_, 1 * sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "find pillar num: " << host_pillar_count[0] << std::endl;

  dim3 mean_block(max_points_in_voxel_, 3);
  PillarMeanFeatureKernel<<<host_pillar_count[0], mean_block, 64 * 3 * sizeof(float)>>>(
      max_voxel_num_,
      max_points_in_voxel_,
      point_feature_dim_,
      dev_num_points_in_voxel, 
      dev_pillar_point_feature,
      dev_points_mean_);

  GatherFeatureKernel<<<max_voxel_num_, max_points_in_voxel_>>>(max_voxel_num_, 
                                                                max_points_in_voxel_, 
                                                                point_feature_dim_,
                                                                min_x_range_, 
                                                                min_y_range_, 
                                                                min_z_range_, 
                                                                pillar_x_size_, 
                                                                pillar_y_size_,
                                                                pillar_z_size_, 
                                                                grid_x_size_, 
                                                                dev_pillar_point_feature,
                                                                dev_num_points_in_voxel, 
                                                                dev_pillar_coors, 
                                                                dev_points_mean_,
                                                                dev_voxel_feature);
}
