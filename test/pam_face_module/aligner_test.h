#ifndef ALIGNER_TEST_H
#define ALIGNER_TEST_H

#include <gtest/gtest.h>
#include <pam_face_module/aligner.h>

class Aligner_test : public ::testing::Test {

public:
  std::unique_ptr<Aligner> aligner_;

  Aligner_test() {}

protected:
  void SetUp() {
    // code here will execute just before the test ensues
  }

  void TearDown() {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }
};

#endif // ALIGNER_TEST_H
