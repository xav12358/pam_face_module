#ifndef ALIGNER_H
#define ALIGNER_H

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

typedef struct {
  Eigen::Matrix2f trans_m;
  Eigen::MatrixXf trans_b;
} Transformation;

class Aligner {

    std::vector<Transformation> image_transformations_;
    std::vector<cv::Mat> dewarped_images_;


    Transformation findTransform(const std::vector<cv::Point2f> input_features,
                                 const std::vector<cv::Point2f> output_features);

public:
  Aligner() {}

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
  void ProcessExtractImages(const cv::Mat &fx1_image);



  //////////
  /// \brief image_transformations
  /// \return
  ///
  std::vector<Transformation> image_transformations() const;


  /////////////////
  /// \brief dewarped_images
  /// \return
  ///
  std::vector<cv::Mat> dewarped_images() const;
};

#endif // ALIGNER_H
