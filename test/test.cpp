#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

#include <gtest/gtest.h>

#include "pam_face_module/aligner_test.h"
#include "pam_face_module/face_detector_test.h"
#include "pam_face_module/face_recognition_test.h"

// TEST_F(Face_detector_test, process) {

//  cv::Mat input_image =
//      cv::imread("../pam_face_module/data/Face1.jpg");
////  cv::resize(input_image, input_image, cv::Size(480, 640), 0, 0);
//  face_detector_.reset(
//      new FaceDetector(input_image.rows, input_image.cols, 40,
//                       "../pam_face_module/data/graph_MTCNN.pb"));
//  face_detector_->Process(input_image);

//  EXPECT_EQ(0, 1);
//}

// TEST_F(Face_recognition_test, process) { EXPECT_EQ(0, 2); }

TEST_F(Aligner_test, process) {
  const float kMeanFaceShape_x[5] = {0.224152, 0.75610125, 0.490127, 0.254149,
                                     0.726104};
  const float kMeanFaceShape_y[5] = {0.2119465, 0.2119465, 0.628106, 0.780233,
                                     0.780233};
  std::vector<cv::Point2f> input_features;
  for (int i = 0; i < 5; i++) {
    cv::Point2f p_input;
    p_input.x = kMeanFaceShape_x[i];
    p_input.y = kMeanFaceShape_y[i];
    input_features.push_back(p_input);
  }
#define DESIRED_SIZE 1.f
  std::vector<std::vector<cv::Point2f>> input_feature_tables;
  input_feature_tables.push_back(input_features);
  aligner_.reset(new Aligner());
  aligner_->ProcessExtractFeatures(DESIRED_SIZE, input_feature_tables);
  std::vector<Transformation> image_transformations =
      aligner_->image_transformations();

#define ABS_ERROR 0.01f
  for (auto t : image_transformations) {
//    std::cout << " //////////////////////// " << std::endl
//              << "trans_b " << std::endl
//              << t.trans_b << std::endl
//              << "trans_m " << std::endl
//              << t.trans_m << std::endl;
    EXPECT_NEAR(t.trans_m(0, 0), 1.f, ABS_ERROR);
    EXPECT_NEAR(t.trans_m(1, 1), 1.f, ABS_ERROR);

    EXPECT_NEAR(t.trans_m(0, 1), 0.f, ABS_ERROR);
    EXPECT_NEAR(t.trans_m(1, 0), 0.f, ABS_ERROR);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
