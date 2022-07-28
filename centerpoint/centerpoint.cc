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

#include "centerpoint.h"

#include <chrono>
#include <cstring>
#include <iostream>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define ANCHOR_NUM 1314144
void CenterPoint::InitParams(const std::string model_config) {
  YAML::Node params = YAML::LoadFile(model_config);
  kPillarXSize =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][0].as<float>();
  kPillarYSize =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][1].as<float>();
  kPillarZSize =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][2].as<float>();
  kMinXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
  kMinYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
  kMinZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
  kMaxXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
  kMaxYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
  kMaxZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
  kNumClass = params["CLASS_NAMES"].size();
  kMaxNumPillars =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_NUMBER_OF_VOXELS"]["test"]
          .as<int>();
  kMaxNumPointsPerPillar =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_POINTS_PER_VOXEL"]
          .as<int>();
  kNumInputBoxFeature = 7;
  kNumOutputBoxFeature = 7;
  kBatchSize = 1;
  kNmsPreMaxsize =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"]
          .as<int>();
  kNmsPostMaxsize =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"]
          .as<int>();
  kPfeChannels = kNumThreads;
  score_threshold_ =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["SCORE_THRESH"]
          .as<float>();
  nms_overlap_threshold_ =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_THRESH"]
          .as<float>();

  // Generate secondary parameters based on above.
  kGridXSize =
      static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize);  // 800
  kGridYSize =
      static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize);  // 800
  kGridZSize = static_cast<int>((kMaxZRange - kMinZRange) / kPillarZSize);  // 1
  assert(1 == kGridZSize);
  kBackboneInputSize = kPfeChannels * kGridYSize * kGridXSize;

  int downsample_ratio = 4;
  kHeadXSize = kGridXSize / downsample_ratio;
  kHeadYSize = kGridYSize / downsample_ratio;
}

CenterPoint::CenterPoint(const bool use_onnx, const std::string pfe_file,
                         const std::string backbone_file,
                         const std::string model_config) {
  InitParams(model_config);
  InitTRT(use_onnx, pfe_file, backbone_file);
  DeviceMemoryMalloc();

  preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
      kNumThreads, kMaxNumPillars, kMaxNumPointsPerPillar, kNumPointFeature,
      kGridXSize, kGridYSize, kGridZSize, kPillarXSize, kPillarYSize,
      kPillarZSize, kMinXRange, kMinYRange, kMinZRange, kMaxXRange, kMaxYRange,
      kMaxZRange));

  scatter_cuda_ptr_.reset(
      new ScatterCuda(kPfeChannels, kGridXSize, kGridYSize));

  const float float_min = std::numeric_limits<float>::lowest();
  const float float_max = std::numeric_limits<float>::max();
  postprocess_cuda_ptr_.reset(new PostprocessCuda(
      kNumThreads, float_min, float_max, kNumClass, kNmsPreMaxsize,
      score_threshold_, nms_overlap_threshold_, kNmsPreMaxsize, kNmsPostMaxsize,
      kNumBoxCorners, kNumInputBoxFeature, kNumOutputBoxFeature));
}

void CenterPoint::DeviceMemoryMalloc() {
  // voxelize
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_num_points_per_pillar_),
                       kMaxNumPillars * sizeof(float)));  // M
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_pillar_point_feature_),
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumPointFeature * sizeof(float)));  // [M , m , 4]
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_pillar_coors_),
                       kMaxNumPillars * 4 * sizeof(int)));  // [M , 4]

  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_pfe_gather_feature_),
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumGatherPointFeature * sizeof(float)));
  // pillar feature encoder
  GPU_CHECK(
      cudaMalloc(&pfe_buffers_[0], kMaxNumPillars * kMaxNumPointsPerPillar *
                                       kNumGatherPointFeature * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[1],
                       kMaxNumPillars * kPfeChannels * sizeof(float)));
  // scatter
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_scattered_feature_),
                       kPfeChannels * kGridYSize * kGridXSize * sizeof(float)));
  // backbone
  GPU_CHECK(
      cudaMalloc(&backbone_buffers_[0], kBackboneInputSize * sizeof(float)));
  /// bbox_preds
  GPU_CHECK(cudaMalloc(&backbone_buffers_[1],
                       18 * kHeadXSize * kHeadYSize * sizeof(float)));
  /// scores
  GPU_CHECK(cudaMalloc(&backbone_buffers_[2],
                       4 * kHeadXSize * kHeadYSize * sizeof(float)));
  /// dir_scores
  GPU_CHECK(cudaMalloc(&backbone_buffers_[3],
                       6 * kHeadXSize * kHeadYSize * sizeof(float)));

  // for filter
  host_box_ = new float[kNmsPreMaxsize * kNumClass * kNumOutputBoxFeature]();
  host_score_ = new float[kNmsPreMaxsize * kNumClass * 18]();
  host_filtered_count_ = new int[kNumClass]();
}

void CenterPoint::SetDeviceMemoryToZero() {
  GPU_CHECK(cudaMemset(dev_num_points_per_pillar_, 0,
                       kMaxNumPillars * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_point_feature_, 0,
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_coors_, 0, kMaxNumPillars * 4 * sizeof(int)));

  GPU_CHECK(cudaMemset(dev_pfe_gather_feature_, 0,
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumGatherPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemset(pfe_buffers_[0], 0,
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumGatherPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemset(pfe_buffers_[1], 0,
                       kMaxNumPillars * kPfeChannels * sizeof(float)));

  GPU_CHECK(cudaMemset(dev_scattered_feature_, 0,
                       kPfeChannels * kGridYSize * kGridXSize * sizeof(float)));

  GPU_CHECK(
      cudaMemset(backbone_buffers_[0], 0, kBackboneInputSize * sizeof(float)));
  GPU_CHECK(cudaMemset(backbone_buffers_[1], 0,
                       18 * kHeadXSize * kHeadYSize * sizeof(float)));
  GPU_CHECK(cudaMemset(backbone_buffers_[2], 0,
                       4 * kHeadXSize * kHeadYSize * sizeof(float)));
  GPU_CHECK(cudaMemset(backbone_buffers_[3], 0,
                       6 * kHeadXSize * kHeadYSize * sizeof(float)));
}

void CenterPoint::InitTRT(const bool use_onnx, const std::string pfe_file,
                          const std::string backbone_file) {
  if (use_onnx) {
    OnnxToTRTModel(pfe_file, &pfe_engine_);
    OnnxToTRTModel(backbone_file, &backbone_engine_);
  } else {
    EngineToTRTModel(pfe_file, &pfe_engine_);
    EngineToTRTModel(backbone_file, &backbone_engine_);
  }
  if (pfe_engine_ == nullptr || backbone_engine_ == nullptr) {
    std::cerr << "Failed to load ONNX file.";
  }

  // create execution context from the engine
  pfe_context_ = pfe_engine_->createExecutionContext();
  backbone_context_ = backbone_engine_->createExecutionContext();
  if (pfe_context_ == nullptr || backbone_context_ == nullptr) {
    std::cerr << "Failed to create TensorRT Execution Context.";
  }
}

void CenterPoint::OnnxToTRTModel(
    const std::string &model_file,  // name of the onnx model
    nvinfer1::ICudaEngine **engine_ptr) {
  std::string model_cache = model_file + ".cache";
  std::fstream trt_cache(model_cache, std::ifstream::in);
  if (!trt_cache.is_open()) {
    std::cout << "Building TRT engine." << std::endl;
    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition *network =
        builder->createNetworkV2(explicit_batch);

    // parse onnx model
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
      std::string msg("failed to parse onnx file");
      g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
      exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(kBatchSize);
    // builder->setHalf2Mode(true);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    bool has_fast_fp16 = builder->platformHasFastFp16();
    if (has_fast_fp16) {
      std::cout << "the platform supports Fp16, use Fp16." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    nvinfer1::ICudaEngine *engine =
        builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }

    // serialize the engine, then close everything down
    auto model_stream = (engine->serialize());
    std::fstream trt_out(model_cache, std::ifstream::out);
    if (!trt_out.is_open()) {
      std::cout << "Can't store trt cache.\n";
      exit(-1);
    }
    trt_out.write((char *)model_stream->data(), model_stream->size());
    trt_out.close();
    model_stream->destroy();

    *engine_ptr = engine;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
  } else {
    std::cout << "Load TRT cache." << std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    trt_cache.seekg(0, trt_cache.end);
    length = trt_cache.tellg();
    trt_cache.seekg(0, trt_cache.beg);

    data = (char *)malloc(length);
    if (data == NULL) {
      std::cout << "Can't malloc data.\n";
      exit(-1);
    }

    trt_cache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(g_logger_);
    if (runtime == nullptr) {
      std::cerr << ": runtime null!" << std::endl;
      exit(-1);
    }
    // plugin_ = nvonnxparser::createPluginFactory(g_logger_);
    nvinfer1::ICudaEngine *engine =
        (runtime->deserializeCudaEngine(data, length, 0));
    if (engine == nullptr) {
      std::cerr << ": engine null!" << std::endl;
      exit(-1);
    }
    *engine_ptr = engine;
    free(data);
    trt_cache.close();
  }
}

void CenterPoint::EngineToTRTModel(const std::string &engine_file,
                                   nvinfer1::ICudaEngine **engine_ptr) {
  int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  std::ifstream cache(engine_file);
  gieModelStream << cache.rdbuf();
  cache.close();
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(g_logger_);

  if (runtime == nullptr) {
    std::string msg("failed to build runtime parser");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();

  gieModelStream.seekg(0, std::ios::beg);
  void *modelMem = malloc(modelSize);
  gieModelStream.read((char *)modelMem, modelSize);

  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  if (engine == nullptr) {
    std::string msg("failed to build engine parser");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  *engine_ptr = engine;

  for (int bi = 0; bi < engine->getNbBindings(); bi++) {
    if (engine->bindingIsInput(bi) == true)
      printf("Binding %d (%s): Input. \n", bi, engine->getBindingName(bi));
    else
      printf("Binding %d (%s): Output. \n", bi, engine->getBindingName(bi));
  }
}

void CenterPoint::DoInference(const float *in_points_array,
                              const int in_num_points,
                              std::vector<float> *out_detections,
                              std::vector<int> *out_labels,
                              std::vector<float> *out_scores) {
  SetDeviceMemoryToZero();
  cudaDeviceSynchronize();
  // [STEP 1] : load pointcloud
  auto load_start = high_resolution_clock::now();
  float *dev_points;
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_points),
                       in_num_points * kNumPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemcpy(dev_points, in_points_array,
                       in_num_points * kNumPointFeature * sizeof(float),
                       cudaMemcpyHostToDevice));
  auto load_end = high_resolution_clock::now();

  // [STEP 2] : preprocess
  auto preprocess_start = high_resolution_clock::now();
  host_pillar_count_[0] = 0;
  preprocess_points_cuda_ptr_->DoPreprocessPointsCuda(
      dev_points, in_num_points, dev_num_points_per_pillar_,
      dev_pillar_point_feature_, dev_pillar_coors_, host_pillar_count_,
      dev_pfe_gather_feature_);
  cudaDeviceSynchronize();
  auto preprocess_end = high_resolution_clock::now();
  // DEVICE_SAVE<float>(
  //     dev_pfe_gather_feature_,
  //     kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature,
  //     "0_Model_pfe_input_gather_feature");

  // [STEP 3] : pfe forward
  cudaStream_t stream;
  GPU_CHECK(cudaStreamCreate(&stream));
  auto pfe_start = high_resolution_clock::now();
  GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[0], dev_pfe_gather_feature_,
                            kMaxNumPillars * kMaxNumPointsPerPillar *
                                kNumGatherPointFeature *
                                sizeof(float),  /// kNumGatherPointFeature
                            cudaMemcpyDeviceToDevice, stream));
  pfe_context_->enqueueV2(pfe_buffers_, stream, nullptr);
  cudaDeviceSynchronize();
  auto pfe_end = high_resolution_clock::now();
  // DEVICE_SAVE<float>(reinterpret_cast<float *>(pfe_buffers_[1]),
  //                    kMaxNumPillars * kPfeChannels,
  //                    "1_Model_pfe_output_buffers_[1]");

  // [STEP 4] : scatter pillar feature
  auto scatter_start = high_resolution_clock::now();
  scatter_cuda_ptr_->DoScatterCuda(host_pillar_count_[0], dev_pillar_coors_,
                                   reinterpret_cast<float *>(pfe_buffers_[1]),
                                   dev_scattered_feature_);
  cudaDeviceSynchronize();
  auto scatter_end = high_resolution_clock::now();
  // DEVICE_SAVE<float>(dev_scattered_feature_, kBackboneInputSize,
  //                    "2_Model_backbone_input_dev_scattered_feature");

  // [STEP 5] : backbone forward
  auto backbone_start = high_resolution_clock::now();
  GPU_CHECK(cudaMemcpyAsync(backbone_buffers_[0], dev_scattered_feature_,
                            kBatchSize * kBackboneInputSize * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
  backbone_context_->enqueueV2(backbone_buffers_, stream, nullptr);
  cudaDeviceSynchronize();
  auto backbone_end = high_resolution_clock::now();
  // DEVICE_SAVE<float>((float *)backbone_buffers_[3], 6 * kHeadXSize * kHeadYSize,
  //                    "00_dir_scores");

  // [STEP 6]: postprocess (multihead)
  auto postprocess_start = high_resolution_clock::now();
  // postprocess_cuda_ptr_->DoPostprocessCuda(
  //     reinterpret_cast<float *>(backbone_buffers_[2]),  // [bboxes]
  //     reinterpret_cast<float *>(backbone_buffers_[1]),  // [scores]
  //     host_box_, host_score_, host_filtered_count_, *out_detections,
  //     *out_labels, *out_scores);
  // cudaDeviceSynchronize();
  auto postprocess_end = high_resolution_clock::now();

  // release the stream and the buffers
  duration<double> preprocess_cost = preprocess_end - preprocess_start;
  duration<double> pfe_cost = pfe_end - pfe_start;
  duration<double> scatter_cost = scatter_end - scatter_start;
  duration<double> backbone_cost = backbone_end - backbone_start;
  duration<double> postprocess_cost = postprocess_end - postprocess_start;
  duration<double> centerpoint_cost = postprocess_end - preprocess_start;
  std::cout << "------------------------------------" << std::endl;
  std::cout << setiosflags(ios::left) << setw(14) << "Module" << setw(12)
            << "Time" << resetiosflags(ios::left) << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::string Modules[] = {"Preprocess", "Pfe",         "Scatter",
                           "Backbone",   "Postprocess", "Summary"};
  double Times[] = {preprocess_cost.count(),  pfe_cost.count(),
                    scatter_cost.count(),     backbone_cost.count(),
                    postprocess_cost.count(), centerpoint_cost.count()};

  for (int i = 0; i < 6; ++i) {
    std::cout << setiosflags(ios::left) << setw(14) << Modules[i] << setw(8)
              << Times[i] * 1000 << " ms" << resetiosflags(ios::left)
              << std::endl;
  }
  std::cout << "------------------------------------" << std::endl;

  cudaStreamDestroy(stream);
  GPU_CHECK(cudaFree(dev_points));
}

CenterPoint::~CenterPoint() {
  // for pillars
  GPU_CHECK(cudaFree(dev_num_points_per_pillar_));
  GPU_CHECK(cudaFree(dev_pillar_point_feature_));
  GPU_CHECK(cudaFree(dev_pillar_coors_));
  // for pfe forward
  GPU_CHECK(cudaFree(dev_pfe_gather_feature_));

  GPU_CHECK(cudaFree(pfe_buffers_[0]));
  GPU_CHECK(cudaFree(pfe_buffers_[1]));

  GPU_CHECK(cudaFree(backbone_buffers_[0]));
  GPU_CHECK(cudaFree(backbone_buffers_[1]));
  GPU_CHECK(cudaFree(backbone_buffers_[2]));

  pfe_context_->destroy();
  backbone_context_->destroy();
  pfe_engine_->destroy();
  backbone_engine_->destroy();
  // for post process
  GPU_CHECK(cudaFree(dev_scattered_feature_));
  delete[] host_box_;
  delete[] host_score_;
  delete[] host_filtered_count_;
}
