#ifndef ALIGNER_TEST_H
#define ALIGNER_TEST_H

#include <gtest/gtest.h>
#include <face_module/aligner.h>

class Aligner_test : public ::testing::Test {

public:
  std::unique_ptr<Aligner> aligner_;

  Aligner_test() {}

  void Setup_ProcessExtractFeatures() {
    const float kMeanFaceShape_x[5] = {0.224152, 0.75610125, 0.490127, 0.254149,
                                       0.726104};
    const float kMeanFaceShape_y[5] = {0.2119465, 0.2119465, 0.628106, 0.780233,
                                       0.780233};
    std::vector<cv::Point2f> input_features;
    for (int i = 0; i < 5; i++) {
      cv::Point2f p_input;
      p_input.x = kMeanFaceShape_x[i];
      p_input.y = kMeanFaceShape_y[i];
      input_features.push_back(p_input);
    }
#define DESIRED_SIZE 1.f
    std::vector<std::vector<cv::Point2f>> input_feature_tables;
    input_feature_tables.push_back(input_features);
    aligner_.reset(new Aligner());
    aligner_->ProcessExtractFeatures(DESIRED_SIZE, input_feature_tables);
    std::vector<Transformation> image_transformations =
        aligner_->image_transformations();
  }

  Transformation Setup_FindTransform() {
    const float kMeanFaceShape_x[5] = {0.224152, 0.75610125, 0.490127, 0.254149,
                                       0.726104};
    const float kMeanFaceShape_y[5] = {0.2119465, 0.2119465, 0.628106, 0.780233,
                                       0.780233};
    std::vector<cv::Point2f> input_features;
    for (int i = 0; i < 5; i++) {
      cv::Point2f p_input;
      p_input.x = kMeanFaceShape_x[i];
      p_input.y = kMeanFaceShape_y[i];
      input_features.push_back(p_input);
    }

    aligner_.reset(new Aligner());
    return  aligner_->FindTransform(input_features, input_features);
  }

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
