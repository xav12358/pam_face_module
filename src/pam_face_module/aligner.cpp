#include "pam_face_module/aligner.h"

Eigen::MatrixXf cv2E(cv::Point2f p) {
  Eigen::MatrixXf ret(1, 2);
  ret(0, 0) = p.x;
  ret(0, 1) = p.y;
  return ret;
}

std::vector<Transformation> Aligner::image_transformations() const {
  return image_transformations_;
}

std::vector<cv::Mat> Aligner::dewarped_images() const {
  return dewarped_images_;
}

Transformation
Aligner::FindTransform(const std::vector<cv::Point2f> input_features,
                       const std::vector<cv::Point2f> output_features) {
  Eigen::Matrix2f cov = Eigen::Matrix2f::Zero(2, 2);

  cv::Point2f input_mean(0, 0), output_mean(0, 0);
  for (int i = 0; i < 5; i++) {
    input_mean += input_features[i];
    output_mean += output_features[i];
  }

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

  Transformation T;
  T.trans_b = trans_b;
  T.trans_m = trans_m;

  return T;
}

void Aligner::ProcessExtractFeatures(
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
  image_transformations_.clear();
  for (auto l : landmarks) {
    image_transformations_.push_back(FindTransform(input_features, l));
  }
}

void Aligner::ProcessExtractImages(const cv::Mat &fx1_image) {}
