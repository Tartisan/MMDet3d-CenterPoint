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

// headers in local files
#include "scatter.h"

__global__ void scatter_kernel(int *dev_pillar_coors, float *pfe_feature,
                               float *scattered_feature, int feature_num,
                               const int grid_x_size, const int grid_y_size) {
  int i_pillar = blockIdx.x;
  int i_feature = threadIdx.x;
  int x_ind = dev_pillar_coors[i_pillar * 4 + 3];
  int y_ind = dev_pillar_coors[i_pillar * 4 + 2];
  float feature = pfe_feature[i_pillar * feature_num + i_feature];
  scattered_feature[i_feature * grid_y_size * grid_x_size +
                    y_ind * grid_x_size + x_ind] = feature;
}

ScatterCuda::ScatterCuda(const int feature_num, const int grid_x_size,
                         const int grid_y_size)
    : feature_num_(feature_num),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size) {}

void ScatterCuda::DoScatterCuda(const int pillar_count, int *dev_pillar_coors,
                                float *pfe_feature, float *scattered_feature) {
  scatter_kernel<<<pillar_count, feature_num_>>>(
      dev_pillar_coors, pfe_feature, scattered_feature, feature_num_,
      grid_x_size_, grid_y_size_);
}
