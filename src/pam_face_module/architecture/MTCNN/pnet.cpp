#include "pam_face_module/architecture/MTCNN/pnet.h"

Pnet::Pnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session,
           int h, int w) {
  graph_ = graph;
  sess_ = session;
  height_ = h;
  width_ = w;

  is_init_ = Init();
}

bool Pnet::Init() {

  // Gerenate input.
  std::shared_ptr<uint8_t> raw_input_data_;
  std::shared_ptr<int64_t> raw_input_dims_;

  // Prepare inputs.
  TF_Output run_inputs_[1];
  std::shared_ptr<TF_Tensor> input_tensor_;
  TF_Tensor *run_inputs_tensors_[1];

  // Generate input.
  {
    raw_input_data_.reset(new uint8_t[3 * height_ * width_],
                           std::default_delete<uint8_t[]>());
    TF_Operation *input_op =
        TF_GraphOperationByName(graph_.get(), "pnet/input");
    run_inputs_[0].oper = input_op;
    run_inputs_[0].index = 0;

    raw_input_dims_.reset(new int64_t[4]{1, height_, width_, 3});
    input_tensor_.reset(
        TF_NewTensor(TF_UINT8, raw_input_dims_.get(), 4, raw_input_data_.get(),
                     3 * height_ * width_ * 3, &nullDeallocator, NULL),
        &TF_DeleteTensor);

    run_inputs_tensors_[0] = input_tensor_.get();
  }

  // Prepare outputs.
  {
    TF_Tensor *run_outputs_tensors_[2];
    TF_Output run_outputs_[2];

    TF_Operation *box_op =
        TF_GraphOperationByName(graph_.get(), "pnet/conv4-2/BiasAdd");
    TF_Operation *score_op = TF_GraphOperationByName(graph_.get(), "pnet/prob1");

    run_outputs_[0].oper = box_op;
    run_outputs_[0].index = 0;

    run_outputs_[1].oper = score_op;
    run_outputs_[1].index = 0;
  }

  return true;
}
