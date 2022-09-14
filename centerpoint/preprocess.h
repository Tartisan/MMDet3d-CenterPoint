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

class PreprocessPointsCuda {
 public:
  PreprocessPointsCuda(const int point_feature_dim,
                       const int max_voxel_num,
                       const int max_points_in_voxel, 
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
                       const float max_z_range);
  ~PreprocessPointsCuda();

  void DoPreprocessPointsCuda(const float* dev_points, 
                              const int in_num_points,
                              int* dev_num_points_in_voxel,
                              float* dev_pillar_point_feature,
                              int* dev_pillar_coors, 
                              int* host_pillar_count,
                              float* dev_voxel_feature);

 private:  
  int max_voxel_num_;
  int max_points_in_voxel_;
  int point_feature_dim_;
  int grid_x_size_;
  int grid_y_size_;
  int grid_z_size_;
  float pillar_x_size_;
  float pillar_y_size_;
  float pillar_z_size_;
  float min_x_range_;
  float min_y_range_;
  float min_z_range_;
  float max_x_range_;
  float max_y_range_;
  float max_z_range_;
  int map_size_;

  float* dev_pillar_point_feature_in_coors_;
  int* pillar_count_histo_;
  int* dev_pillar_num_;
  float* dev_points_mean_;

  const int kNumThreads = 64;
};