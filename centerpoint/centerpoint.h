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

// headers in STL
#include <assert.h>
#include <yaml-cpp/yaml.h>

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
// headers in TensorRT
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
  CenterPoint(const bool use_onnx, const std::string pfe_file,
              const std::string backbone_file, const std::string model_config);

  ~CenterPoint();

  void DoInference(const float *in_points_array, const int in_num_points,
                   std::vector<float> *out_detections,
                   std::vector<int> *out_labels,
                   std::vector<float> *out_scores);

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
  // voxel size
  float kPillarXSize;
  float kPillarYSize;
  float kPillarZSize;
  // point cloud range
  float kMinXRange;
  float kMinYRange;
  float kMinZRange;
  float kMaxXRange;
  float kMaxYRange;
  float kMaxZRange;
  // hyper parameters
  int kNumClass;
  int kMaxNumPillars;
  int kMaxNumPointsPerPillar;
  int kNumPointFeature = 4;  // [x, y, z, i]
  int kNumAnchorSize = 7;
  // if you need to change this, watch the gather_point_feature_kernel func in
  // preprocess
  int kNumThreads = 64; // also used for number of scattered feature
  int kNumGatherPointFeature = 10;
  int kGridXSize;
  int kGridYSize;
  int kGridZSize;
  int kPfeChannels;
  int kBackboneInputSize;
  int kNumInputBoxFeature;
  int kNumOutputBoxFeature;
  int kBatchSize;
  int kNumBoxCorners = 8;
  int kNmsPreMaxsize;
  int kNmsPostMaxsize;

  // preprocess
  int host_pillar_count_[1];
  float *dev_num_points_per_pillar_;
  float *dev_pillar_point_feature_;
  int *dev_pillar_coors_;
  float *dev_pfe_gather_feature_;
  // pfe
  void *pfe_buffers_[2];
  // scatter
  float *dev_scattered_feature_;
  // backbone
  int kHeadXSize;
  int kHeadYSize;
  void *backbone_buffers_[4];
  // postprocess
  float score_threshold_;
  float nms_overlap_threshold_;
  float *host_box_;
  float *host_score_;
  int *host_filtered_count_;

  std::unique_ptr<PreprocessPointsCuda> preprocess_points_cuda_ptr_;
  std::unique_ptr<ScatterCuda> scatter_cuda_ptr_;
  std::unique_ptr<PostprocessCuda> postprocess_cuda_ptr_;

  Logger g_logger_;
  nvinfer1::ICudaEngine *pfe_engine_;
  nvinfer1::ICudaEngine *backbone_engine_;
  nvinfer1::IExecutionContext *pfe_context_;
  nvinfer1::IExecutionContext *backbone_context_;
};
