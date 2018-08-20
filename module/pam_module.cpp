#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include <tensorflow/c/c_api.h>

//#include "face_module/architecture/FaceNet/facenet.h"

#include "face_module/feature_detector.h"

int main(int argc, char **argv) {

    cv::Mat input_image = cv::imread("../../pam_face_module/test/data/Face1.jpg");
//    std::unique_ptr<Aligner> aligner;
//    aligner.reset(new Aligner());

    cv::imshow("iii",input_image);
    cv::waitKey(-1);


    std::unique_ptr<FeatureDetector> feature_detector(new FeatureDetector());

    return 0;
}
