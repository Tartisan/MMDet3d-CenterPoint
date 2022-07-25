// headers in STL
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>

#include "centerpoint/centerpoint.h"

void Getinfo(void) {
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
    printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
           prop.maxGridSize[2]);
  }
  printf("\n");
}

int Txt2Arrary(float *&points_array, std::string file_name,
               int num_feature = 4) {
  std::ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  std::vector<float> temp_points;
  std::string c;

  while (!InFile.eof()) {
    InFile >> c;
    temp_points.push_back(atof(c.c_str()));
  }
  points_array = new float[temp_points.size()];
  for (size_t i = 0; i < temp_points.size(); ++i) {
    points_array[i] = temp_points[i];
  }

  InFile.close();
  return temp_points.size() / num_feature;
  // printf("Done");
};

int Bin2Arrary(float *&points_array, std::string file_name,
               int in_num_feature = 4, int out_num_feature = 4) {
  std::ifstream InFile;
  InFile.open(file_name.data(), ios::binary);
  assert(InFile.is_open());
  std::vector<float> temp_points;
  float f;

  while (!InFile.eof()) {
    InFile.read((char *)&f, sizeof(f));
    temp_points.push_back(f);
  }
  points_array = new float[temp_points.size()];
  int size = temp_points.size() / in_num_feature;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < out_num_feature; ++j) {
      points_array[i * out_num_feature + j] =
          temp_points[i * in_num_feature + j];
    }
  }

  InFile.close();
  return size;
  // printf("Done");
};

void Boxes2Txt(const std::vector<float> &boxes, 
               const std::vector<int> &labels, 
               const std::vector<float> &scores, 
               std::string file_name, 
               int num_feature = 7) {
  std::ofstream ofFile;
  ofFile.open(file_name, std::ios::out);
  if (ofFile.is_open()) {
    for (size_t i = 0; i < boxes.size() / num_feature; ++i) {
      for (int j = 0; j < num_feature; ++j) {
        ofFile << boxes.at(i * num_feature + j) << " ";
      }
      ofFile << labels.at(i) << " ";
      ofFile << scores.at(i) << " ";
      ofFile << "\n";
    }
  }
  ofFile.close();
  return;
};

void load_anchors(float *&anchor_data, std::string file_name) {
  std::ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  std::vector<float> temp_points;
  std::string c;

  while (!InFile.eof()) {
    InFile >> c;
    temp_points.push_back(atof(c.c_str()));
  }
  anchor_data = new float[temp_points.size()];
  for (size_t i = 0; i < temp_points.size(); ++i) {
    anchor_data[i] = temp_points[i];
  }
  InFile.close();
  return;
}

void test(void) {
  const std::string DB_CONF = "../bootstrap.yaml";
  YAML::Node config = YAML::LoadFile(DB_CONF);
  // std::string pfe_file, backbone_file;
  // pfe_file = config["PfeOnnx"].as<std::string>();
  // backbone_file = config["BackboneOnnx"].as<std::string>();
  // std::cout << "pfe_file: " << pfe_file << std::endl;
  // std::cout << "backbone_file: " << backbone_file << std::endl;
  std::string model_file = config["ModelFile"].as<std::string>();
  std::cout << "modle_file: " << modle_file << std::endl;

  const std::string model_config = config["ModelConfig"].as<std::string>();
  std::string file_name = config["InputFile"].as<std::string>();
  std::cout << "config: " << model_config << std::endl;
  std::cout << "data: " << file_name << std::endl;

  CenterPoint cp(config["UseOnnx"].as<bool>(), model_file, model_config);

  float *points_array;
  int in_num_points;
  in_num_points =
      Bin2Arrary(points_array, file_name, config["LoadDim"].as<int>(),
                 config["UseDim"].as<int>());
  std::cout << "num points: " << in_num_points << std::endl;

  for (int _ = 0; _ < 1; _++) {
    std::vector<float> out_detections;
    std::vector<int> out_labels;
    std::vector<float> out_scores;

    cudaDeviceSynchronize();
    cp.DoInference(points_array, in_num_points, &out_detections, &out_labels,
                   &out_scores);
    cudaDeviceSynchronize();
    size_t BoxFeature = 7;
    size_t num_objects = out_detections.size() / BoxFeature;
    std::cout << "detected objects: " << num_objects << std::endl;
    assert(num_objects == out_labels.size());
    assert(num_objects == out_scores.size());

    std::string boxes_file_name = config["OutputFile"].as<std::string>();
    Boxes2Txt(out_detections, out_labels, out_scores, boxes_file_name);
  }

  delete[] points_array;
};

int main(int argc, char **argv) {
  Getinfo();
  test();
}
