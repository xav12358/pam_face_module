#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

#include "pam_face_module/face_detector_test.h"
#include "pam_face_module/face_recognition_test.h"

TEST_F(Face_detector_test, process) { EXPECT_EQ(0, 1); }

TEST_F(Face_recognition_test, process) { EXPECT_EQ(0, 2); }


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
