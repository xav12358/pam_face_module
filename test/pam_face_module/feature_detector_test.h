#ifndef FEATURE_DETECTOR_TEST_H
#define FEATURE_DETECTOR_TEST_H


#include <gtest/gtest.h>
#include <pam_face_module/feature_detector.h>


class Feature_detector_test : public ::testing::Test {
public:
  std::unique_ptr<FeatureDetector> feature_detector_;

  Feature_detector_test() {}

protected:

  ////////////////
  /// \brief Face_detector::init
  /// \param fileName
  ///
  bool Init(const std::string filename) {
      return feature_detector_->Init(filename);
  }

  ////////////////
  /// \brief Normalize
  /// \param u8x3_Image
  /// \return
  ///
  cv::Mat Normalize(cv::Mat u8x3_Image) {
      return feature_detector_->Normalize(u8x3_Image);
  }

  //////////////////
  /// \brief Process
  /// \param
  ///
  void Process(std::vector<cv::Mat> &image_candidates) {
      feature_detector_->Process(image_candidates);
  }


  void SetUp() {
    // code here will execute just before the test ensues
  }

  void TearDown() {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }
};

#endif // FEATURE_DETECTOR_TEST_H
