#include "pam_face_module/face_feature.h"

FaceFeature::FaceFeature(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) : Net(graph,session){
}



bool FaceFeature::Init() {
    // Prepare input
    {
        TF_Operation *input_op = TF_GraphOperationByName(graph_.get(), "input");
        run_inputs_[0].oper = input_op;
        run_inputs_[0].index = 0;
    }

    // Prepare outputs.
    {
        TF_Operation *feature_op = TF_GraphOperationByName(graph_.get(), "l2_normalize");
        run_outputs_[0].oper = feature_op;
        run_outputs_[0].index = 0;

    }
    is_init_ = true;
    return is_init_;
}


void FaceFeature::Process( std::vector<cv::Mat> &image_candidates) {
    Debug(" >>>>> FaceRecognition::Process ");
//    int batch = int(image_candidates.size());

//    /* prepare input image data */
//    int input_size = batch * kHeight_ * kWidth_ * 3;
//    std::vector<float> input_buffer((size_t)(input_size));
//    float *input_data = input_buffer.data();

//    int patch_size = kWidth_ * kHeight_ * 3;
//    for (size_t i = 0; i < image_candidates.size(); i++) {
//        copy_one_patch(img, image_candidates[i], input_data, kHeight_, kWidth_);
//        input_data += patch_size;
//    }

//    // Generate input.
//    std::shared_ptr<int64_t> raw_input_dims_;
//    raw_input_dims_.reset(new int64_t[4]{batch, kHeight_, kWidth_, 3});
//    std::shared_ptr<TF_Tensor> input_tensor_;
//    input_tensor_.reset(
//        TF_NewTensor(TF_FLOAT, raw_input_dims_.get(), 4, input_buffer.data(),
//                     sizeof(float) * batch * kHeight_ * kWidth_ * 3, &nullDeallocator, NULL),
//        &TF_DeleteTensor);

//    run_inputs_tensors_[0] = input_tensor_.get();

//    TF_SessionRun(sess_.get(), nullptr, run_inputs_, run_inputs_tensors_, 1, run_outputs_,
//                  &run_output_tensors_[0], 2, nullptr, 0, nullptr, status_.get());

//    if (TF_GetCode(status_.get()) != TF_OK) {
//        std::cerr << "ERROR: Unable to run output_op: " << std::string(TF_Message(status_.get()))
//                  << std::endl;
//        return;
//    }

//    /*retrieval the forward results*/
//    final_boxes_.clear();
//    const float *conf_data = (const float *)(TF_TensorData(run_output_tensors_[1]));
//    const float *reg_data = (const float *)(TF_TensorData(run_output_tensors_[0]));

//    for (size_t i = 0; i < batch; i++) {
//        if (conf_data[1] > kThreshold) {
//            FaceBox output_box;
//            FaceBox &input_box = pnet_candidates[i];
//            output_box.x0 = input_box.x0;
//            output_box.y0 = input_box.y0;
//            output_box.x1 = input_box.x1;
//            output_box.y1 = input_box.y1;

//            output_box.score = *(conf_data + 1);

//            /*Note: regress's value is swaped here!!!*/
//            output_box.regress[0] = reg_data[1];
//            output_box.regress[1] = reg_data[0];
//            output_box.regress[2] = reg_data[3];
//            output_box.regress[3] = reg_data[2];
//            final_boxes_.push_back(output_box);
//        }
//        conf_data += 2;
//        reg_data += 4;
//    }

    TF_DeleteTensor(run_output_tensors_[0]);
    TF_DeleteTensor(run_output_tensors_[1]);
}
