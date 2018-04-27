#ifndef FACE_RECOGNITION_TEST_H
#define FACE_RECOGNITION_TEST_H

#include <gtest/gtest.h>
#include <pam_face_module/face_recognition.h>

class Face_recognition_test : public ::testing::Test {

 public:
    Face_recognition_test();

 protected:
    void SetUp() {
        // code here will execute just before the test ensues
    }

    void TearDown() {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }
};

#endif  // FACE_RECOGNITION_TEST_H
