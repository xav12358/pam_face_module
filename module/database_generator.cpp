#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

#include "face_module/face_detector.h"
#include "face_module/feature_detector.h"
#include "face_module/utils/parser.h"

void processCapture(std::string input_stream, std::string output_feature_folder,
                    std::string graph_MTCNN, std::string graph_FaceFeature) {

  if (IsNumber(input_stream)) {
    int index_cam = std::stoi(input_stream, nullptr, 10);
    cv::VideoCapture video_capture(index_cam);

    bool complete = false;
    cv::Mat frame;
    video_capture >> frame;
    std::unique_ptr<FaceDetector> face_detector(
        new FaceDetector(frame.rows, frame.cols, 40));
    if (!face_detector->Init(graph_MTCNN)) {
      std::cerr << "Can init the face_detector" << std::endl;
    }

    std::unique_ptr<FeatureDetector> feature_detector(new FeatureDetector());
    if (feature_detector->Init(graph_FaceFeature)) {
      std::cerr << "Can init the feature_detector" << std::endl;
    }

    while (!complete) {
      // grab image
      video_capture >> frame;

      // Detect the face in the current image
      face_detector->Process(frame);
      auto face_boxes = face_detector->face_list();

      // For each face box align it and exact the warped region
      for(auto box: face_boxes){

      }
    }
  }
}

int main(int argc, char **argv) {
  std::unordered_map<std::string, std::string> parsed_command =
      ParseCommand(argc, argv);

  if (!(parsed_command.count("--input_stream") &&
        parsed_command.count("--output_folder") &&
        parsed_command.count("--graph_MTCNN") &&
        parsed_command.count("--graph_FaceFeature"))) {
    std::cout << "Help command:" << std::endl
              << " database_generator "
                 "--input_stream index_cam"
                 "--output_folder folder"
                 "--graph_MTCNN graph_MTCNN.pb"
                 "--graph_FaceFeature graph_FaceFeature.pb"
              << std::endl;
  }

  std::string input_stream = parsed_command["--input_stream"];
  std::string output_feature_folder = parsed_command["--output_folder"];
  std::string graph_MTCNN = parsed_command["--graph_MTCNN"];
  std::string graph_FaceFeature = parsed_command["--graph_FaceFeature"];

  return 0;
}
