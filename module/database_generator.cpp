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
    cv::VideoCapture *video_capture = new cv::VideoCapture(0);

    if (!video_capture->isOpened()) {
      std::cerr << "Can't open camera " << index_cam << std::endl;
    }

    video_capture->set(CV_CAP_PROP_FRAME_WIDTH, 640);
    video_capture->set(CV_CAP_PROP_FRAME_HEIGHT, 480 );

    bool complete = false;
    cv::Mat u8x3_image = cv::imread("/home/xavier/Desktop/image.png");
//    *video_capture >> u8x3_image;

    std::unique_ptr<FaceDetector> face_detector(
        new FaceDetector(u8x3_image.rows, u8x3_image.cols, 80 ));
    if (!face_detector->Init(graph_MTCNN)) {
      std::cerr << "Can't init the face_detector" << std::endl;
    }

//    std::unique_ptr<FeatureDetector> feature_detector(new FeatureDetector());
//    if (feature_detector->Init(graph_FaceFeature)) {
//      std::cerr << "Can't init the feature_detector" << std::endl;
//    }

    while (!complete) {
      // grab image
//      *video_capture >> u8x3_image;
        u8x3_image = cv::imread("/home/xavier/Desktop/image.png");

      // Detect the face in the current image
      face_detector->Process(u8x3_image);
      auto face_boxes = face_detector->face_list();

      // For each face box align it and exact the warped region
      for (auto f : face_boxes) {
        cv::Rect r(f.px0, f.py0, f.px1 - f.px0, f.py1 - f.py0);
        cv::rectangle(u8x3_image, r, cv::Scalar(0, 255, 0), 2);
        for (int j = 0; j < 5; j++) {
          cv::circle(u8x3_image, cv::Point(f.landmark.x[j], f.landmark.y[j]),
                     10, cv::Scalar(0, 0, 255), 5);
          std::cout << "landmark " << j << " "
                    << cv::Point(f.landmark.x[j], f.landmark.y[j]) << std::endl;
        }
      }

      std::cout << "lalal" << u8x3_image.empty() << std::endl;
      cv::imshow("facedetector", u8x3_image);
      cv::waitKey(10);
    }
  }
}

int main(int argc, char **argv) {

    std::unordered_map<std::string, std::string> parsed_command =
        ParseCommand(argc, argv);

    for (auto p : parsed_command) {
      std::cout << " p " << p.first << " " << p.second << std::endl;
    }

    std::cout << "parsed_command.count(--input_stream) "
              << parsed_command.count("--input_stream") << std::endl;
    std::cout << "parsed_command.count(--output_folder) "
              << parsed_command.count("--output_folder") << std::endl;
    std::cout << "parsed_command.count(--graph_MTCNN) "
              << parsed_command.count("--graph_MTCNN") << std::endl;
    std::cout << "parsed_command.count(--graph_FaceFeature) "
              << parsed_command.count("--graph_FaceFeature") << std::endl;
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
//  std::string input_stream = "0";
//  std::string output_feature_folder = "";
//  std::string graph_MTCNN = "/home/xavier/Desktop/developpement/Network/"
//                            "pam_face_module/test/data/graph/graph_MTCNN.pb";
//  std::string graph_FaceFeature = "/home/xavier/Desktop/developpement/Network/"
//                                  "pam_face_module/test/data/graph/"
//                                  "graph_FaceFeature.pb";

  //  cv::VideoCapture *video_capture = new cv::VideoCapture(0);

  //  if (!video_capture->isOpened()) {
  //      std::cerr << "Can't open camera " << 0 << std::endl;
  //  }
  ////  cv::Mat input_image =
  /// cv::imread("../../pam_face_module/test/data/warped_images/chips0.png");

  //  while(1) {
  //      cv::Mat u8x3_image;

  //      *video_capture >> u8x3_image;
  //      std::cout << "lalal"<< u8x3_image.empty() << std::endl;
  //      cv::imshow("facedetector", u8x3_image);
  //      cv::waitKey(100);
  //  }

  processCapture(input_stream, output_feature_folder, graph_MTCNN,
                 graph_FaceFeature);

  return 0;
}
