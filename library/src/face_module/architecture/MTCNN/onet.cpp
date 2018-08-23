#include "face_module/architecture/MTCNN/onet.h"

std::vector<FaceBox> Onet::final_boxes() const { return final_boxes_; }

Onet::Onet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) : Net(graph,session){
}

bool Onet::Init() {
    // Prepare input
    {
        TF_Operation *input_op = TF_GraphOperationByName(graph_.get(), "onet/input");
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
    is_init_ = true;
    return is_init_;
}

void Onet::Process(cv::Mat &img, std::vector<FaceBox> &rnet_candidates) {
    Debug(" >>>>> Onet::Process ");
    int batch = int(rnet_candidates.size());

    /* prepare input image data */
    int input_size = batch * kHeight_ * kWidth_ * 3;
    std::vector<float> input_buffer((size_t)(input_size));
    float *input_data = input_buffer.data();

    int patch_size = kWidth_ * kHeight_ * 3;
    for (size_t i = 0; i < rnet_candidates.size(); i++) {
        copy_one_patch(img, rnet_candidates[i], input_data, kHeight_, kWidth_);
        input_data += patch_size;
    }

    // Generate input.
    std::shared_ptr<int64_t> raw_input_dims_;
    raw_input_dims_.reset(new int64_t[4]{batch, kHeight_, kWidth_, 3});
    std::shared_ptr<TF_Tensor> input_tensor_;
    input_tensor_.reset(
        TF_NewTensor(TF_FLOAT, raw_input_dims_.get(), 4, input_buffer.data(),
                     sizeof(float) * batch * kHeight_ * kWidth_ * 3, &nullDeallocator, NULL),
        &TF_DeleteTensor);

    run_inputs_tensors_[0] = input_tensor_.get();

    TF_SessionRun(sess_.get(), nullptr, run_inputs_, run_inputs_tensors_, 1, run_outputs_,
                  &run_output_tensors_[0], 3, nullptr, 0, nullptr, status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
        std::cerr << "ERROR: Unable to run output_op: " << std::string(TF_Message(status_.get()))
                  << std::endl;
        return;
    }

    /*retrieval the forward results*/
    final_boxes_.clear();
    const float *conf_data = (const float *)TF_TensorData(run_output_tensors_[2]);
    const float *reg_data = (const float *)TF_TensorData(run_output_tensors_[0]);
    const float *points_data = (const float *)TF_TensorData(run_output_tensors_[1]);

    for (int i = 0; i < batch; i++) {
        if (conf_data[1] > kThreshold) {
            FaceBox output_box;
            FaceBox &input_box = rnet_candidates[i];

            output_box.x0 = input_box.x0;
            output_box.y0 = input_box.y0;
            output_box.x1 = input_box.x1;
            output_box.y1 = input_box.y1;

            output_box.px0 = input_box.px0;
            output_box.py0 = input_box.py0;
            output_box.px1 = input_box.px1;
            output_box.py1 = input_box.py1;

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
        conf_data += 2;
        reg_data += 4;
        points_data += 10;
    }
}
