#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <tensorflow/c/c_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

#include "pam_face_module/face_detector_test.h"
#include "pam_face_module/face_recognition_test.h"

TEST_F(Face_detector_test, process) {
    cv::Mat input_image =
        //          cv::imread("/home/xavier/Desktop/developpement/Network/pam_face_module/"
        //                     "data/2018-05-21-152745.jpg")
        //      cv::imread("/home/xavier/Desktop/developpement/Network/pam_face_module/"
        //                 "data/2018-03-25-102906.jpg");
        cv::imread(
            "../pam_face_module/"
            "data/Face1.jpg");

    cv::resize(input_image, input_image, cv::Size(160, 120), 0, 0);
    cv::imshow("tt", input_image);
    cv::waitKey(-1);

    std::shared_ptr<FaceDetector> lface_detector(new FaceDetector(
        input_image.rows, input_image.cols, 40, "../pam_face_module/data/graph_MTCNN.pb"));
    lface_detector->Process(input_image);
    EXPECT_EQ(0, 1);
}

TEST_F(Face_recognition_test, process) { EXPECT_EQ(0, 2); }

int main(int argc, char **argv) {

//    std::vector<float> tab;
//    tab.reserve(48*48*3*1000);

//    int64 timestart = cv::getCPUTickCount();
//    int nbiter = 100;
//    for (int i = 0;i< nbiter;i++){
//           std::vector<float> tab;
//           tab.reserve(48*48*3*1000);
//    }
//    int64 timestop = cv::getCPUTickCount();

//    float ms = (timestop - timestart) / cv::getTickFrequency() * 1000.0;

//    std::cout << "ms " << ms /(float) nbiter <<std::endl;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
