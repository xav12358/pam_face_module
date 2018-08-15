#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

#include <gtest/gtest.h>

#include "face_module/aligner_test.h"
#include "face_module/face_detector_test.h"
#include "face_module/feature_manager_test.h"
#include "face_module/feature_detector_test.h"

#include "face_module/utils/parser.h"

//TEST(ParserCommand, parse){
//    char *argv[] = {"myexe","--input", "lala", "gg", "--output", "lolo", "hhh"};
//    int argc = 7;
//    std::unordered_map<std::string,std::string> parsedCommand = ParseCommand(argc,argv);
//    EXPECT_EQ(parsedCommand["--input"] , "lala");
//    EXPECT_EQ(parsedCommand["--output"] , "lolo");
//}
///////////////////////////////////////////

//TEST_F(Feature_detector_test, Init) {
//    feature_detector_.reset(new FeatureDetector());
//    EXPECT_TRUE(Init("../../pam_face_module/test/data/graph/graph_FaceFeature.pb"));
//}

//TEST_F(Feature_detector_test, Normalize) {
//    feature_detector_.reset(new FeatureDetector());
//    Init("../../pam_face_module/test/data/graph/graph_FaceFeature.pb");
//    cv::Mat input_image = cv::imread("../../pam_face_module/test/data/warped_images/chips0.png");
//    cv::Mat normalized_image = Normalize(input_image);

//}

//TEST_F(Feature_detector_test, Process) {
//    feature_detector_.reset(new FeatureDetector());
//    Init("../../pam_face_module/test/data/graph/graph_FaceFeature.pb");
//    cv::Mat input_image = cv::imread("../../pam_face_module/test/data/warped_images/chips0.png");
//    std::vector<cv::Mat> raw_image_vector;
//    raw_image_vector.push_back(input_image);
//    feature_detector_->Process(raw_image_vector);
//}


///////////////////////////////////////////


TEST_F(Face_detector_test, LoadGraph) {
    face_detector_.reset(new FaceDetector(100, 100, 40));
    //  face_detector_->Init();
    EXPECT_TRUE(LoadGraph("../../pam_face_module/test/data/graph/graph_MTCNN.pb"));
}

TEST_F(Face_detector_test, CreateSession) {
    face_detector_.reset(new FaceDetector(100, 100, 40));
    //  face_detector_->Init();
    EXPECT_TRUE(CreateSession());
}

TEST_F(Face_detector_test, CreateArchitecture) {
    face_detector_.reset(new FaceDetector(100, 100, 40));
    //  face_detector_->Init();
    LoadGraph("../../pam_face_module/test/data/graph/graph_MTCNN.pb");
    CreateSession();
    EXPECT_TRUE(CreateArchitecture());
}


#include <numeric>
#include <chrono>

TEST_F(Face_detector_test, process) {
    cv::Mat input_image = cv::imread("../../pam_face_module/test/data/Face1.jpg");
    face_detector_.reset(
                new FaceDetector(input_image.rows, input_image.cols, 40));
    face_detector_->Init("../../pam_face_module/test/data/graph/graph_MTCNN.pb");

    // Warm up
    face_detector_->Process(input_image);

    // start the record
    auto c_start = std::chrono::high_resolution_clock::now();;
    for(int i =0;i<10;i++){
        face_detector_->Process(input_image);
    }
    std::cout << "Time to execute :" << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - c_start).count()/10 << std::endl;

    EXPECT_EQ(0, 1);
}

///////////////////////////////////////////
//TEST_F(Aligner_test, test_ProcessExtractFeature) {
//    Setup_ProcessExtractFeatures();
//#define ABS_ERROR 0.01f
//    for (auto t : aligner_->image_transformations()) {
//        EXPECT_NEAR(t.trans_m(0, 0), 1.f, ABS_ERROR);
//        EXPECT_NEAR(t.trans_m(1, 1), 1.f, ABS_ERROR);
//        EXPECT_NEAR(t.trans_m(0, 1), 0.f, ABS_ERROR);
//        EXPECT_NEAR(t.trans_m(1, 0), 0.f, ABS_ERROR);
//    }
//}

//TEST_F(Aligner_test, test_FindTransform) {
//    Transformation t = Setup_FindTransform();
//#define ABS_ERROR 0.01f
//    EXPECT_NEAR(t.trans_m(0, 0), 1.f, ABS_ERROR);
//    EXPECT_NEAR(t.trans_m(1, 1), 1.f, ABS_ERROR);
//    EXPECT_NEAR(t.trans_m(0, 1), 0.f, ABS_ERROR);
//    EXPECT_NEAR(t.trans_m(1, 0), 0.f, ABS_ERROR);
//}

///////////////////////////////////////////

//TEST_F(Feature_manager_test, test_read) {
//    auto dataset =
//            FeatureManager::Read("../../pam_face_module/test/data/feature_file/facerec_128D.txt");
//    auto dataset_iter = dataset.begin();
//    EXPECT_TRUE(dataset.size() == 2);
//    EXPECT_TRUE((*dataset_iter).first == "DavidVu");
//}

//TEST_F(Feature_manager_test, test_write) {
//    auto dataset =
//            FeatureManager::Read("../../pam_face_module/test/data/feature_file/facerec_128D.txt");
//    auto dataset_iter = dataset.begin();

//    std::string filename_tmp =
//            "../../pam_face_module/test/data/feature_file/tmp_facerec_128D.txt";
//    FeatureManager::Write(dataset, filename_tmp);

//    auto dataset_tmp = FeatureManager::Read(filename_tmp);
//    auto dataset_iter_tmp = dataset_tmp.begin();

//    EXPECT_TRUE(dataset_tmp.size() == 2);
//    EXPECT_TRUE((*dataset_iter_tmp).first == "DavidVu");

//    std::remove(filename_tmp.c_str());
//}

#include <face_module/utils/parser.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
