#include "pam_face_module/architecture/MTCNN/pnet.h"

Pnet::Pnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session, int h, int w,
           float scale)
    : Net(graph, session) {
    height_ = h;
    width_ = w;
    scale_ = scale;

    is_init_ = Init();
}

std::vector<FaceBox> Pnet::final_candidate_boxes() const { return final_candidate_boxes_; }

void Pnet::FeedML(cv::Mat const &u8x3_image) {
    cv::MatConstIterator_<cv::Vec3f> iter = u8x3_image.begin<cv::Vec3f>();
    cv::MatConstIterator_<cv::Vec3f> iter_end = u8x3_image.end<cv::Vec3f>();

    for (; iter != iter_end; ++iter) {
        cv::Vec3f pixel = *iter;
        cv::Point pos = iter.pos();

        int pt = 3 * (pos.x + pos.y * width_);
        raw_input_data_.get()[pt] = pixel[0];
        raw_input_data_.get()[pt + 1] = pixel[1];
        raw_input_data_.get()[pt + 2] = pixel[2];
    }
}

void Pnet::FetchData() {
    /*retrieval the forward results*/
    const float *conf_data = (const float *)TF_TensorData(run_output_tensors_[1]);
    const float *reg_data = (const float *)TF_TensorData(run_output_tensors_[0]);

    int feature_h = int(TF_Dim(run_output_tensors_[0], 1));
    int feature_w = int(TF_Dim(run_output_tensors_[0], 2));

    int64_t conf_size = feature_h * feature_w * 2;

    std::vector<FaceBox> candidate_boxes;
    GenerateBoundingBox(conf_data, conf_size, reg_data, scale_, kPnetThreshold, feature_h,
                        feature_w, candidate_boxes, true);

    final_candidate_boxes_.clear();
    nms_boxes(candidate_boxes, 0.5, NMSType::kNMS_UNION, final_candidate_boxes_);
}

void Pnet::GenerateBoundingBox(const float *confidence_data, int confidence_size,
                               const float *reg_data, float scale, float threshold, int feature_h,
                               int feature_w, std::vector<FaceBox> &output, bool transposed) {
    int stride = 2;
    int cellSize = 12;

    int img_h = feature_h;
    int img_w = feature_w;

    for (int y = 0; y < img_h; y++)
        for (int x = 0; x < img_w; x++) {
            int line_size = img_w * 2;
            float score = confidence_data[line_size * y + 2 * x + 1];

            if (score >= threshold) {
                float top_x = (int)((x * stride + 1) / scale);
                float top_y = (int)((y * stride + 1) / scale);
                float bottom_x = (int)((x * stride + cellSize) / scale);
                float bottom_y = (int)((y * stride + cellSize) / scale);

                FaceBox box;
                box.x0 = top_x;
                box.y0 = top_y;
                box.x1 = bottom_x;
                box.y1 = bottom_y;

                box.score = score;

                int c_offset = (img_w * 4) * y + 4 * x;

                if (transposed) {
                    box.regress[1] = reg_data[c_offset];
                    box.regress[0] = reg_data[c_offset + 1];
                    box.regress[3] = reg_data[c_offset + 2];
                    box.regress[2] = reg_data[c_offset + 3];
                } else {
                    box.regress[0] = reg_data[c_offset];
                    box.regress[1] = reg_data[c_offset + 1];
                    box.regress[2] = reg_data[c_offset + 2];
                    box.regress[3] = reg_data[c_offset + 3];
                }

                output.push_back(box);
            }
        }
}

bool Pnet::Init() {
    // Generate input.
    {
        raw_input_data_.reset(new float[3 * height_ * width_], std::default_delete<float[]>());
        TF_Operation *input_op = TF_GraphOperationByName(graph_.get(), "pnet/input");
        run_inputs_[0].oper = input_op;
        run_inputs_[0].index = 0;

        raw_input_dims_.reset(new int64_t[4]{1, height_, width_, 3});
        input_tensor_.reset(
            TF_NewTensor(TF_FLOAT, raw_input_dims_.get(), 4, raw_input_data_.get(),
                         sizeof(float) * height_ * width_ * 3, &nullDeallocator, NULL),
            &TF_DeleteTensor);

        run_inputs_tensors_[0] = input_tensor_.get();
    }

    // Prepare outputs.
    {
        TF_Operation *box_op = TF_GraphOperationByName(graph_.get(), "pnet/conv4-2/BiasAdd");
        TF_Operation *score_op = TF_GraphOperationByName(graph_.get(), "pnet/prob1");

        run_outputs_[0].oper = box_op;
        run_outputs_[0].index = 0;

        run_outputs_[1].oper = score_op;
        run_outputs_[1].index = 0;
    }

    status_.reset(TF_NewStatus(), std::default_delete<TF_Status>());

    return true;
}

void Pnet::Process(cv::Mat const &fx3_image) {
    Debug(" >>>>> Onet::Process ");

    // Put the data to the input placeholder
    FeedML(fx3_image);

    // run the session
    RunSession();

    // Fetch data
    FetchData();
}

void Pnet::RunSession() {
    // run session
    TF_SessionRun(sess_.get(), NULL, run_inputs_, run_inputs_tensors_, 1, run_outputs_,
                  &run_output_tensors_[0], 2, NULL, 0, NULL, status_.get());
    if (TF_GetCode(status_.get()) != TF_OK) {
        std::cerr << "ERROR: Unable to run output_op: " << std::string(TF_Message(status_.get()))
                  << std::endl;
        return;
    }
}
