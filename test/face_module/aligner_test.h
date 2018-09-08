#ifndef ALIGNER_TEST_H
#define ALIGNER_TEST_H

#include <gtest/gtest.h>
#include <face_module/aligner.h>

class Aligner_test : public ::testing::Test {

public:
  std::unique_ptr<Aligner> aligner_;

  Aligner_test() {}


  cv::Mat Setup_FindTransform(std::vector<cv::Point2f> landmarks) {
      float image_size = 160;

      const float kMeanFaceShape_x[5] = {0.224152, 0.75610125, 0.490127, 0.254149,
                                         0.726104};
      const float kMeanFaceShape_y[5] = {0.2119465, 0.2119465, 0.628106, 0.780233,
                                         0.780233};
      const float kPadding = 0.1f;

      std::vector<cv::Point2f> input_features;
      for (int i = 0; i < 5; i++) {
        cv::Point2f p_input;
        p_input.x =
            (kPadding + kMeanFaceShape_x[i]) / (2.f * kPadding + 1.f) * image_size;
        p_input.y =
            (kPadding + kMeanFaceShape_y[i]) / (2.f * kPadding + 1.f) * image_size;
        input_features.push_back(p_input);
      }

    aligner_.reset(new Aligner());
    return aligner_->FindTransform(landmarks,input_features , image_size);
  }

  std::vector<std::pair<cv::Mat, std::string>>  Setup_ProcessExtractImages(cv::Mat u8x3_image, std::vector<FaceBox> facebox_list) {
    return  aligner_->ProcessExtractImages(u8x3_image, facebox_list, Aligner::SMALL_ORIENTATION);
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
