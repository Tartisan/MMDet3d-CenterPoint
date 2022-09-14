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

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "common.h"
#include "postprocess.h"
#include "preprocess.h"
#include "scatter.h"

using namespace std;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING)
      : reportable_severity(severity) {}

  void log(Severity severity, const char *msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportable_severity;
};

class CenterPoint {
 public:
  CenterPoint(const YAML::Node &config, const std::string pfe_file,
              const std::string backbone_file, const std::string model_config);

  ~CenterPoint();

  void DoInference(const float *in_points_array, const int in_num_points,
                   std::vector<Box> &out_detections);

 private:
  void DeviceMemoryMalloc();

  void SetDeviceMemoryToZero();

  void InitParams(const std::string model_config);

  void InitTRT(const bool use_onnx, const std::string pfe_file,
               const std::string backbone_file);

  void OnnxToTRTModel(const std::string &model_file,
                      nvinfer1::ICudaEngine **engine_ptr);

  void EngineToTRTModel(const std::string &engine_file,
                        nvinfer1::ICudaEngine **engine_ptr);

  void Preprocess(const float *in_points_array, const int in_num_points);

 private:
  bool enable_debug_ = false;
  std::string trt_mode_ = "fp32";

  const int kBatchSize = 1;
  const int point_feature_dim_ = 4;  // [x, y, z, i]
  const int voxel_feature_dim_ = 10;
  const int pillar_feature_dim_ = 64;
  
  float pillar_x_size_;
  float pillar_y_size_;
  float pillar_z_size_;
  float min_x_range_;
  float min_y_range_;
  float min_z_range_;
  float max_x_range_;
  float max_y_range_;
  float max_z_range_;
  int grid_x_size_;
  int grid_y_size_;
  int grid_z_size_;
  int num_classes_;
  int max_voxel_num_;
  int max_points_in_voxel_;

  int backbone_input_size_;
  int downsample_size_;
  int head_x_size_;
  int head_y_size_;
  int nms_pre_max_size_;
  int nms_post_max_size_;
  float score_thresh_;
  float nms_overlap_thresh_;

  int backbone_map_size_;
  int head_map_size_;
  std::map<std::string, int> head_map_;
  std::vector<int> num_classes_in_task_;
  int num_tasks_;
  int box_range_;
  int cls_range_;
  int dir_range_;

  int host_pillar_count_[1];
  int *dev_num_points_in_voxel_;
  float *dev_pillar_point_feature_;
  int *dev_pillar_coors_;
  float *dev_voxel_feature_;
  void *pfe_buffers_[2];
  float *dev_canvas_feature_;
  void *backbone_buffers_[4];

  std::unique_ptr<PreprocessPointsCuda> preprocess_points_cuda_ptr_;
  std::unique_ptr<ScatterCuda> scatter_cuda_ptr_;
  std::unique_ptr<PostprocessCuda> postprocess_cuda_ptr_;

  Logger g_logger_;
  nvinfer1::ICudaEngine *pfe_engine_;
  nvinfer1::ICudaEngine *backbone_engine_;
  nvinfer1::IExecutionContext *pfe_context_;
  nvinfer1::IExecutionContext *backbone_context_;
};
