#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <memory>
#include <tensorflow/c/c_api.h>

#include <pam_face_module/architecture/FaceNet/facenet.h>
#include <pam_face_module/utils.h>

class Feature_detector_test;
class FeatureDetector {

    friend class Feature_detector_test;

    // FeatureDetector features
    bool is_init_;

    // Architecture
    std::shared_ptr<FaceNet> facenet_;
public:
  FeatureDetector();

  ////////////////
  /// \brief Face_detector::init
  /// \param fileName
  ///
  bool Init(const std::string filename);

  ////////////////
  /// \brief Normalize
  /// \param u8x3_Image
  /// \return
  ///
  cv::Mat Normalize(cv::Mat u8x3_Image);


  //////////////////
  /// \brief Process
  /// \param
  ///
  void Process(std::vector<cv::Mat> &image_candidates);

};

#endif // FACE_DETECTOR_H
