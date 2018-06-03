#ifndef FACE_DETECTOR_TEST_H
#define FACE_DETECTOR_TEST_H

#include <memory>
#include <gtest/gtest.h>
#include <pam_face_module/face_detector.h>

class Face_detector_test : public ::testing::Test {

 public:
    std::unique_ptr< FaceDetector> face_detector_;

    Face_detector_test(){}

    ~Face_detector_test() {}

 protected:
    void SetUp() {
        // code here will execute just before the test ensues
    }

    void TearDown() {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }
};

#endif  // FACE_DETECTOR_TEST_H
