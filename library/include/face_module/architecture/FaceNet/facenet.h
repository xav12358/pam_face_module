#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H

#include <face_module/architecture/MTCNN/net.h>

#include <eigen3/Eigen/Dense>

class FaceNet : public Net {

  const int kHeight_ = 160;
  const int kWidth_ = 160;
  const int kFeatureSize_ = 128;

  //     Gerenate input.
  std::shared_ptr<float> raw_input_data_;
  std::shared_ptr<int64_t> raw_input_dims_;

  // Prepare inputs.
  TF_Output run_inputs_[1];
  std::shared_ptr<TF_Tensor> input_tensor_;
  TF_Tensor *run_inputs_tensors_[1];

  // Generate output
  TF_Output run_outputs_[1];
  TF_Tensor *run_output_tensors_[1];

public:
  FaceNet();


  //////////////////////
  /// \brief CreateArchitecture
  /// \return
  ///
  bool CreateArchitecture();

  /////////////
  /// \brief createSession
  /// \return
  ///
  bool CreateSession();


  ////////////////////
  /// \brief Face_detector::loadGraph
  /// \param fileName
  ///
  bool LoadGraph(std::string const fileName);


  ////////////////////
  /// \brief Init
  /// \return
  ///
  bool Init();

  /////////////////
  /// \brief Init
  /// \param filename
  /// \return
  ///
  bool Init(std::string const filename);

  ////////////////////
  /// \brief Process
  /// \param image_candidates
  ///
  void Process( std::vector<cv::Mat> &image_candidates);
};

#endif // FACE_RECOGNITION_H
