#ifndef ALIGNER_H
#define ALIGNER_H

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>



class Aligner {

    std::vector<cv::Mat> image_transformations_;
    std::vector<cv::Mat> dewarped_images_;



public:
  Aligner() {}


  //////////////////////
  /// \brief findTransform fibd transforma between input_features and output_features
  /// \param input_features
  /// \param output_features
  /// \return
  ///
  cv::Mat FindTransform(const std::vector<cv::Point2f> input_features,
                               const std::vector<cv::Point2f> output_features,
                               float desired_size);
  ///////////////////
  /// \brief ProcessExtractFeatures Extract features transfrmation for each facebox
  /// \param image_size
  /// \param landmarks
  ///
  void ProcessExtractFeatures(float image_size,
                                const std::vector<std::vector<cv::Point2f> > landmarks);

  /////////////////
  /// \brief ProcessExtractImages Extract aligned images containing facebox
  /// \param fx1_image
  ///
  std::vector<cv::Mat> ProcessExtractImages(const cv::Mat &u8x1_image,
                            const std::vector<std::vector<cv::Point2f> > landmarks);



  //////////
  /// \brief image_transformations
  /// \return
  ///
  std::vector<cv::Mat> image_transformations() const;


  /////////////////
  /// \brief dewarped_images
  /// \return
  ///
  std::vector<cv::Mat> dewarped_images() const;
};

#endif // ALIGNER_H
