#include "pam_face_module/architecture/MTCNN/onet.h"

Onet::Onet(std::shared_ptr<TF_Graph> graph,
           std::shared_ptr<TF_Session> session) {
  graph_ = graph;
  sess_ = session;
  is_init_ = Init();
}

bool Onet::Init() {
  // Prepare input
  {
    TF_Operation *input_op =
        TF_GraphOperationByName(graph_.get(), "onet/input");
    run_inputs_[0].oper = input_op;
    run_inputs_[0].index = 0;
  }

  // Prepare outputs.
  {
    TF_Operation *box_op =
        TF_GraphOperationByName(graph_.get(), "onet/conv6-2/conv6-2");
    TF_Operation *feature_op =
        TF_GraphOperationByName(graph_.get(), "onet/conv6-3/conv6-3");
    TF_Operation *score_op =
        TF_GraphOperationByName(graph_.get(), "onet/prob1");

    run_outputs_[0].oper = box_op;
    run_outputs_[0].index = 0;

    run_outputs_[1].oper = feature_op;
    run_outputs_[1].index = 0;

    run_outputs_[2].oper = score_op;
    run_outputs_[2].index = 0;
  }

  status_.reset(TF_NewStatus(), std::default_delete<TF_Status>());
}

void Onet::Process(cv::Mat &img, std::vector<FaceBox> &rnet_candidates) {
  Debug("Onet::Process " + std::to_string(rnet_candidates.size()));
  int batch = int(rnet_candidates.size());

  /* prepare input image data */
  int input_size = batch * kHeight_ * kWidth_ * 3;
  std::vector<float> input_buffer((size_t)(input_size));
  float *input_data = input_buffer.data();

  for (size_t i = 0; i < rnet_candidates.size(); i++) {
    int patch_size = kWidth_ * kHeight_ * 3;
    copy_one_patch(img, rnet_candidates[i], input_data, kHeight_, kWidth_);
    input_data += patch_size;
  }

  // Generate input.
  std::shared_ptr<int64_t> raw_input_dims_;
  raw_input_dims_.reset(new int64_t[4]{batch, kHeight_, kWidth_, 3});
  std::shared_ptr<TF_Tensor> input_tensor_;
  input_tensor_.reset(TF_NewTensor(TF_FLOAT, raw_input_dims_.get(), 4,
                                   input_buffer.data(),
                                   sizeof(float) * kHeight_ * kWidth_ * 3,
                                   &nullDeallocator, NULL),
                      &TF_DeleteTensor);

  run_inputs_tensors_[0] = input_tensor_.get();

  TF_SessionRun(sess_.get(), nullptr, run_inputs_, run_inputs_tensors_, 1,
                run_outputs_, &run_output_tensors_[0], 2, nullptr, 0, nullptr,
                status_.get());

  if (TF_GetCode(status_.get()) != TF_OK) {
    std::cerr << "ERROR: Unable to run output_op: "
              << std::string(TF_Message(status_.get())) << std::endl;
    return;
  }

  /*retrieval the forward results*/
  final_boxes_.clear();
  const float *conf_data = (const float *)TF_TensorData(run_output_tensors_[2]);
  const float *reg_data = (const float *)TF_TensorData(run_output_tensors_[0]);
  const float *points_data =
      (const float *)TF_TensorData(run_output_tensors_[1]);

  for (int i = 0; i < batch; i++) {
    if (conf_data[1] > kThreshold) {
      FaceBox output_box;

      FaceBox &input_box = rnet_candidates[i];

      output_box.x0 = input_box.x0;
      output_box.y0 = input_box.y0;
      output_box.x1 = input_box.x1;
      output_box.y1 = input_box.y1;

      output_box.score = conf_data[1];

      output_box.regress[0] = reg_data[1];
      output_box.regress[1] = reg_data[0];
      output_box.regress[2] = reg_data[3];
      output_box.regress[3] = reg_data[2];

      /*Note: switched x,y points value too..*/
      for (int j = 0; j < 5; j++) {
        output_box.landmark.x[j] = *(points_data + j + 5);
        output_box.landmark.y[j] = *(points_data + j);
      }

      final_boxes_.push_back(output_box);
    }


    Debug( "Onet::Process final " + std::to_string(final_boxes_.size()));

    conf_data += 2;
    reg_data += 4;
    points_data += 10;
  }
}
