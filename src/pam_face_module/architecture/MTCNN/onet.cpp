#include "pam_face_module/architecture/MTCNN/onet.h"

Onet::Onet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) {
    graph_ = graph;
    sess_ = session;
    is_init_ = Init();
}

bool Onet::Init() {
    // Prepare input
    {
        TF_Operation *input_op = TF_GraphOperationByName(graph_.get(), "rnet/input");
        run_inputs_[0].oper = input_op;
        run_inputs_[0].index = 0;
    }

    // Prepare outputs.
    {
        TF_Operation *box_op = TF_GraphOperationByName(graph_.get(), "onet/conv6-2/conv6-2");
        TF_Operation *feature_op = TF_GraphOperationByName(graph_.get(), "onet/conv6-3/conv6-3");
        TF_Operation *score_op = TF_GraphOperationByName(graph_.get(), "onet/prob1");

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
    Debug("Rnet::Process " + std::to_string(rnet_candidates.size()));
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
    input_tensor_.reset(
        TF_NewTensor(TF_FLOAT, raw_input_dims_.get(), 4, input_buffer.data(),
                     sizeof(float) * kHeight_ * kWidth_ * 3, &nullDeallocator, NULL),
        &TF_DeleteTensor);

    run_inputs_tensors_[0] = input_tensor_.get();

    TF_SessionRun(sess_.get(), nullptr, run_inputs_, run_inputs_tensors_, 1, run_outputs_,
                  &run_output_tensors_[0], 2, nullptr, 0, nullptr, status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
        std::cerr << "ERROR: Unable to run output_op: " << std::string(TF_Message(status_.get()))
                  << std::endl;
        return;
    }

}
