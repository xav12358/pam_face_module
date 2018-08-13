#ifndef FEATURE_MANAGER_TEST_H
#define FEATURE_MANAGER_TEST_H


#include <gtest/gtest.h>
#include <face_module/feature_manager.h>

class Feature_manager_test : public ::testing::Test {

public:
  Feature_manager_test() {}

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
