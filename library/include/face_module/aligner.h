#ifndef ALIGNER_H
#define ALIGNER_H

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "face_module/architecture/FaceNet/facenet.h"


class Aligner {

//    std::vector<cv::Mat> image_transformations_;
//    std::vector<cv::Mat> dewarped_images_;

public:    
    typedef enum {
        NONE,
        TOP_LEFT,
        TOP_HCENTER,
        TOP_RIGHT,
        VCENTER_LEFT,
        VCENTER_HCENTER,
        VCENTER_RIGHT,
        BOTTOM_LEFT,
        BOTTOM_HCENTER,
        BOTTOM_RIGHT,
    }FaceOrientation;

    Aligner() {}



    static std::string toString(FaceOrientation f){
        switch(f){
        case NONE:
            return "NONE";
            break;
        case TOP_LEFT:
            return "TOP_LEFT";
            break;
        case TOP_HCENTER:
            return "TOP_HCENTER";
            break;
        case TOP_RIGHT:
            return "TOP_RIGHT";
            break;
        case VCENTER_LEFT:
            return "VCENTER_LEFT";
            break;
        case VCENTER_HCENTER:
            return "VCENTER_HCENTER";
            break;
        case VCENTER_RIGHT:
            return "VCENTER_RIGHT";
            break;
        case BOTTOM_LEFT:
            return "BOTTOM_LEFT";
            break;
        case BOTTOM_HCENTER:
            return "BOTTOM_HCENTER";
            break;
        case BOTTOM_RIGHT:
            return "BOTTOM_RIGHT";
            break;
        }
    }

  //////////////////////
  /// \brief findTransform fibd transforma between input_features and output_features
  /// \param input_features
  /// \param output_features
  /// \return
  ///
  cv::Mat FindTransform(const std::vector<cv::Point2f> input_features,
                               const std::vector<cv::Point2f> output_features,
                               float desired_size);


  /////////////
  /// \brief GetPosition
  ///
  std::string GetPosition(const std::vector<cv::Point2f> landmarks);

  ///////////////////
  /// \brief ProcessExtractFeatures Extract features transfrmation for each facebox
  /// \param image_size
  /// \param landmarks
  ///
  std::vector<cv::Mat> ProcessExtractFeatures(float image_size,
                                const std::vector<std::vector<cv::Point2f> > landmarks);

  /////////////////
  /// \brief ProcessExtractImages Extract aligned images containing facebox
  /// \param fx1_image
  ///
  std::vector<std::pair<cv::Mat, std::string> > ProcessExtractImages(const cv::Mat u8x3_image,
                            const std::vector<FaceBox> box_list);



  //////////
  /// \brief image_transformations
  /// \return
  ///
  std::vector<cv::Mat> image_transformations() const;


  /////////////////
  /// \brief dewarped_images
  /// \return
  ///
  std::vector<std::pair<cv::Mat, FaceOrientation>> dewarped_images() const;
};

#endif // ALIGNER_H
