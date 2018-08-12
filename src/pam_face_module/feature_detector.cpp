#include "pam_face_module/feature_detector.h"

#include <iostream>

FeatureDetector::FeatureDetector() {}

bool FeatureDetector::Init(std::string const filename) {
  facenet_ = std::make_shared<FaceNet>();
  is_init_ = facenet_->Init(filename);
  return is_init_;
}

cv::Mat FeatureDetector::Normalize(cv::Mat u8x3_Image) {

  int kHeight_ = u8x3_Image.rows;
  int kWidth_ = u8x3_Image.cols;
  // Process the RGB 160x160 image, get the mean and stdev value to normalize it
  cv::Mat u8x1_Image = cv::Mat(kHeight_ * 3, kWidth_, CV_8UC1, u8x3_Image.data);

  cv::Mat mean;
  cv::Mat stddev;
  cv::meanStdDev(u8x1_Image, mean, stddev);

  double d_mean = mean.at<double>(0, 0), d_stddev = stddev.at<double>(0, 0);
  double d_std_adj =
      std::max(d_stddev, (double)(1 / (float(kHeight_) * float(kWidth_))));

  cv::Mat fx3_Image;
  u8x3_Image.convertTo(fx3_Image, CV_32FC3);
  fx3_Image = (fx3_Image - d_mean) / d_std_adj;

  return fx3_Image;
}

void FeatureDetector::Process(std::vector<cv::Mat> &image_candidates) {
  std::vector<cv::Mat> normalized_images;
  for (auto image : image_candidates) {
    normalized_images.push_back(Normalize(image));
  }
  facenet_->Process(normalized_images);
}
