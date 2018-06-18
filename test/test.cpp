#include <cstdio>
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
#include "pam_face_module/face_feature_test.h"
#include "pam_face_module/feature_manager_test.h"

 TEST_F(Face_detector_test, LoadGraph) {
  face_detector_.reset(new FaceDetector(100, 100, 40));
//  face_detector_->Init();
  EXPECT_TRUE(LoadGraph("../pam_face_module/test/data/graph_MTCNN.pb"));
}

 TEST_F(Face_detector_test, CreateSession) {
  face_detector_.reset(new FaceDetector(100, 100, 40));
//  face_detector_->Init();
  EXPECT_TRUE(CreateSession());
}

 TEST_F(Face_detector_test, CreateArchitecture) {
  face_detector_.reset(new FaceDetector(100, 100, 40));
//  face_detector_->Init();
  LoadGraph("../pam_face_module/test/data/graph_MTCNN.pb");
  CreateSession();
  EXPECT_TRUE(CreateArchitecture());
}

 TEST_F(Face_detector_test, process) {
  cv::Mat input_image = cv::imread("../pam_face_module/test/data/Face1.jpg");
  face_detector_.reset(
      new FaceDetector(input_image.rows, input_image.cols, 40 ));
  face_detector_->Init("../pam_face_module/test/data/graph_MTCNN.pb");
  face_detector_->Process(input_image);
  EXPECT_EQ(0, 1);
}

 TEST_F(Aligner_test, test_ProcessExtractFeature) {

  Setup_ProcessExtractFeatures();

#define ABS_ERROR 0.01f
  for (auto t : aligner_->image_transformations()) {
    EXPECT_NEAR(t.trans_m(0, 0), 1.f, ABS_ERROR);
    EXPECT_NEAR(t.trans_m(1, 1), 1.f, ABS_ERROR);

    EXPECT_NEAR(t.trans_m(0, 1), 0.f, ABS_ERROR);
    EXPECT_NEAR(t.trans_m(1, 0), 0.f, ABS_ERROR);
  }
}

 TEST_F(Aligner_test, test_FindTransform) {

  Transformation t = Setup_FindTransform();
#define ABS_ERROR 0.01f
  EXPECT_NEAR(t.trans_m(0, 0), 1.f, ABS_ERROR);
  EXPECT_NEAR(t.trans_m(1, 1), 1.f, ABS_ERROR);

  EXPECT_NEAR(t.trans_m(0, 1), 0.f, ABS_ERROR);
  EXPECT_NEAR(t.trans_m(1, 0), 0.f, ABS_ERROR);
}

TEST_F(Feature_manager_test, test_read) {
  auto dataset =
      FeatureManager::Read("../pam_face_module/test/data/facerec_128D.txt");
  auto dataset_iter = dataset.begin();
  EXPECT_TRUE(dataset.size() == 2);
  EXPECT_TRUE((*dataset_iter).first == "DavidVu");
}

TEST_F(Feature_manager_test, test_write) {
  auto dataset =
      FeatureManager::Read("../pam_face_module/test/data/facerec_128D.txt");
  auto dataset_iter = dataset.begin();

  std::string filename_tmp =
      "../pam_face_module/test/data/tmp_facerec_128D.txt";
  FeatureManager::Write(dataset, filename_tmp);

  auto dataset_tmp = FeatureManager::Read(filename_tmp);
  auto dataset_iter_tmp = dataset_tmp.begin();

  EXPECT_TRUE(dataset_tmp.size() == 2);
  EXPECT_TRUE((*dataset_iter_tmp).first == "DavidVu");

  std::remove(filename_tmp.c_str());
}

int main(int argc, char **argv) {

  //    cv::Mat imager = cv::Mat::ones(cv::Size(10,10),CV_32F) *1.0;
  //    cv::Mat imageg = cv::Mat::ones(cv::Size(10,10),CV_32F) *2.0;
  //    cv::Mat imageb = cv::Mat::ones(cv::Size(10,10),CV_32F) *4.0;
  //    std::vector<cv::Mat> channels;

  //    cv::Mat fin_img;

  //    channels.push_back(imager);
  //    channels.push_back(imageg);
  //    channels.push_back(imageb);
  //    merge(channels, fin_img);

  //    cv::Mat     mean;
  //    cv::Mat     stddev;

  //    cv::meanStdDev ( fin_img, mean, stddev );

  //    std::cout  << "mean.size "<< mean.size() << " " <<
  //    mean.at<double>(0,0)<< mean.at<double>(1,0)<< mean.at<double>(2,0)<<
  //    std::endl;

  //    std::cout  << "mean.size "<< stddev.size() << std::endl;

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
