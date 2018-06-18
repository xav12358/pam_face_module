#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H

#include <pam_face_module/architecture/MTCNN/net.h>

class FaceNet : public Net {

  const int kHeight_ = 160;
  const int kWidth_ = 160;

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
  FaceNet(std::shared_ptr<TF_Graph> graph,
                  std::shared_ptr<TF_Session> session);

  bool Init();


  void Process( std::vector<cv::Mat> &image_candidates);
};

#endif // FACE_RECOGNITION_H
