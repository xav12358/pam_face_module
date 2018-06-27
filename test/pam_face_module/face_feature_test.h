#ifndef FACE_FEATURE_TEST_H
#define FACE_FEATURE_TEST_H

#include <gtest/gtest.h>
#include <pam_face_module/architecture/FaceNet/facenet.h>

class FaceNet_test : public ::testing::Test {

public:
  FaceNet_test () {}

protected:
  void SetUp() {
    // code here will execute just before the test ensues
  }

  void TearDown() {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }
};

#endif // FACE_FEATURE_TEST_H
