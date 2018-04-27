#ifndef FACE_DETECTOR_TEST_H
#define FACE_DETECTOR_TEST_H

#include <gtest/gtest.h>
#include <pam_face_module/face_detector.h>

class Face_detector_test : public ::testing::Test {
    Face_detector face_detector_;

 public:
    Face_detector_test() {}
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
