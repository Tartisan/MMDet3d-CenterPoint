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
  PreprocessPointsCuda(const int num_threads, const int num_point_feature,
                       const int max_num_pillars,
                       const int max_points_per_pillar, const int grid_x_size,
                       const int grid_y_size, const int grid_z_size,
                       const float pillar_x_size, const float pillar_y_size,
                       const float pillar_z_size, const float min_x_range,
                       const float min_y_range, const float min_z_range,
                       const float max_x_range, const float max_y_range,
                       const float max_z_range);
  ~PreprocessPointsCuda();

  void DoPreprocessPointsCuda(const float* dev_points, const int in_num_points,
                              float* dev_num_points_per_pillar,
                              float* dev_pillar_point_feature,
                              int* dev_pillar_coors, int* host_pillar_count,
                              float* dev_pfe_gather_feature);

 private:
  const int kNumThreads_;
  const int kMaxNumPillars_;
  const int kMaxNumPointsPerPillar_;
  const int kNumPointFeature_;
  const int kGridXSize_;
  const int kGridYSize_;
  const int kGridZSize_;
  const float kPillarXSize_;
  const float kPillarYSize_;
  const float kPillarZSize_;
  const float kMinXRange_;
  const float kMinYRange_;
  const float kMinZRange_;
  const float kMaxXRange_;
  const float kMaxYRange_;
  const float kMaxZRange_;

  float* dev_pillar_point_feature_in_coors_;
  int* mask_;
  int* dev_pillar_count_;
  float* dev_points_mean_;
};