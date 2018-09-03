#include "face_module/aligner.h"

#include <math.h>

Eigen::MatrixXf cv2E(cv::Point2f p) {
  Eigen::MatrixXf ret(1, 2);
  ret(0, 0) = p.x;
  ret(0, 1) = p.y;
  return ret;
}

// std::vector<cv::Mat> Aligner::image_transformations() const {
//  return image_transformations_;
//}

// std::vector<std::pair<cv::Mat, Aligner::FacePosition>>
// Aligner::dewarped_images() const {
//  return dewarped_images_;
//}

cv::Mat Aligner::FindTransform(const std::vector<cv::Point2f> input_features,
                               const std::vector<cv::Point2f> output_features,
                               float desired_size) {
  Eigen::Matrix2f cov = Eigen::Matrix2f::Zero(2, 2);

  cv::Point2f input_mean(0, 0), output_mean(0, 0);
  for (int i = 0; i < 5; i++) {
    input_mean += input_features[i];
    output_mean += output_features[i];
  }
  input_mean /= 5.0;
  output_mean /= 5.0;

  Eigen::MatrixXf m1x2_input_mean;
  Eigen::MatrixXf m1x2_output_mean;
  m1x2_input_mean = cv2E(input_mean);
  m1x2_output_mean = cv2E(output_mean);

  float input_sigma = 0, output_sigma = 0;
  for (int i = 0; i < 5; i++) {
    Eigen::MatrixXf m1x2_input;
    Eigen::MatrixXf m1x2_output;
    m1x2_input = cv2E(input_features[i]);
    m1x2_output = cv2E(output_features[i]);

    m1x2_input -= m1x2_input_mean;
    m1x2_output -= m1x2_output_mean;

    float dis = m1x2_input.norm();
    input_sigma += dis * dis;
    dis = m1x2_output.norm();
    output_sigma += dis * dis;

    cov += m1x2_input.transpose() * m1x2_output;
  }

  input_sigma /= 5.0f;
  output_sigma /= 5.0f;
  cov /= 5.0f;

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);

  Eigen::MatrixXf S = Eigen::Matrix2f::Identity(2, 2);
  Eigen::MatrixXf d = svd.singularValues();
  Eigen::MatrixXf U = svd.matrixU();
  Eigen::MatrixXf Vt = svd.matrixV();

  if (cov.determinant() < 0) {
    if (d(1, 0) < d(0, 0)) {
      S(1, 1) = -1.f;
    } else {
      S(0, 0) = -1.f;
    }
  }

  Eigen::MatrixXf d_diag = Eigen::Matrix2f::Identity(2, 2);
  d_diag(0, 0) = d(0, 0);
  d_diag(1, 1) = d(1, 0);

  Eigen::MatrixXf R = U * S * Vt;
  float c = 1.0f;
  if (input_sigma != 0)
    c = 1.f / input_sigma * (d_diag * S).trace();

  Eigen::MatrixXf trans_b =
      m1x2_output_mean.transpose() - c * R * m1x2_input_mean.transpose();

  Eigen::MatrixXf trans_m = c * R;

  float angle = -180.0 / M_PI * std::atan2(trans_m(0, 1), trans_m(0, 0));
  float scale = (trans_m.block<1, 2>(0, 0)).norm();

  Eigen::MatrixXf input_center(2, 1);
  input_center(0, 0) = (input_features[0].x + input_features[1].x) / 2.0;
  input_center(1, 0) = (input_features[0].y + input_features[1].y) / 2.0;

  Eigen::MatrixXf output_center(2, 1);
  output_center(0, 0) = desired_size * 0.5f;
  output_center(1, 0) = desired_size * 0.4f;

  Eigen::MatrixXf e_xy = output_center - input_center;
  cv::Mat rot = cv::getRotationMatrix2D(
      cv::Point2f(input_center(0, 0), input_center(1, 0)), angle, scale);

  rot.at<double>(0, 2) += e_xy(0, 0);
  rot.at<double>(1, 2) += e_xy(1, 0);

  return rot;
}

std::string Aligner::GetPosition(const std::vector<cv::Point2f> landmarks) {

  float y0 = (landmarks[0].y + landmarks[1].y) / 2;
  float y1 = landmarks[2].y;
  float y2 = (landmarks[3].y + landmarks[4].y) / 2;

  //    if (fabs((y0 - y1) / (y1 - y2)) > 1.35  ) {
  //      // Bottom
  //      return toString(BOTTOM_RIGHT);
  //    } else if ((fabs(y1 - y0) / fabs(y1 - y2)) < 0.65) {
  //      // Top
  //      return toString(TOP_RIGHT);
  //    } else {
  //      // Center V
  //      return toString(VCENTER_RIGHT);
  //    }



  if ((fabs(landmarks[0].x - landmarks[2].x) /
       fabs(landmarks[1].x - landmarks[2].x)) > 2) {
    // Right H
    if (fabs((y0 - y1) / (y1 - y2)) > 1.35) {
      // Bottom
      return toString(BOTTOM_RIGHT);
    } else if ((fabs(y1 - y0) / fabs(y1 - y2)) < 0.65) {
      // Top
      return toString(TOP_RIGHT);
    } else {
      // Center V
      return toString(VCENTER_RIGHT);
    }
  } else if ((fabs(landmarks[1].x - landmarks[2].x) /
                  fabs(landmarks[0].x - landmarks[2].x) >
              2)) {
    // LEFT H
    if (fabs((y0 - y1) / (y1 - y2)) > 1.35) {
      // Bottom
      return toString(BOTTOM_LEFT);
    } else if ((fabs(y1 - y0) / fabs(y1 - y2)) < 0.65) {
      // Top
      return toString(TOP_LEFT);
    } else {
      // Center V
      return toString(VCENTER_LEFT);
    }
  } else {
    // Center H
      if (fabs((y0 - y1) / (y1 - y2)) > 1.35) {
        // Bottom
        return toString(BOTTOM_HCENTER);
      } else if ((fabs(y1 - y0) / fabs(y1 - y2)) < 0.65) {
        // Top
        return toString(TOP_HCENTER);
      } else {
        // Center V
        return toString(VCENTER_HCENTER);
      }
  }

}

std::vector<cv::Mat> Aligner::ProcessExtractFeatures(
    float image_size, const std::vector<std::vector<cv::Point2f>> landmarks) {
  // Face features
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

  // Extract transformation from the reference landmarks positions
  std::vector<cv::Mat> image_transformations;
  for (auto l : landmarks) {
    image_transformations.push_back(
        FindTransform(l, input_features, image_size));
  }
  return image_transformations;
}

std::vector<std::pair<cv::Mat, std::string>>
Aligner::ProcessExtractImages(const cv::Mat u8x3_image,
                              const std::vector<FaceBox> box_list) {

  std::vector<std::vector<cv::Point2f>> landmark_list;
  for (auto box : box_list) {
    std::vector<cv::Point2f> landmarks;
    for (int i = 0; i < 5; i++) {
      landmarks.push_back(cv::Point2f(box.landmark.x[i], box.landmark.y[i]));
    }
    landmark_list.push_back(landmarks);
  }

  const int image_size = 160;
  auto image_transformations =
      ProcessExtractFeatures(image_size, landmark_list);

  // Get warped images
  std::vector<std::pair<cv::Mat, std::string>> image_patch_and_orientation;
  for (auto t_rot : image_transformations) {
    cv::Mat patch;
    cv::warpAffine(u8x3_image, patch, t_rot, cv::Size(image_size, image_size));
    image_patch_and_orientation.push_back(
        std::pair<cv::Mat, std::string>(patch, "NONE"));
  }

  for (int i = 0; i < box_list.size(); i++) {
    image_patch_and_orientation.at(i).second = GetPosition(landmark_list[i]);
  }

#ifdef USEDEBUG
//  cv::namedWindow("u8x1_image", cv::WINDOW_NORMAL);
//  cv::imshow("u8x1_image", u8x1_image);
//  cv::imshow("patch", patch);
//  cv::waitKey(-1);
#endif

  return image_patch_and_orientation;
}
