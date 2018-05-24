#include "pam_face_module/face_detector.h"

#include <iostream>

FaceDetector::FaceDetector(int h, int w, int min_size, std::string const finename)
    : height_(h), width_(w), min_size_(min_size) {
    is_init_ = Init(finename);
}

bool FaceDetector::CreateArchitecture() {
    // Generate the pnet pyramidal stages

    int min_side = std::min(height_, width_);
    float m = 12.0f / min_size_;

    min_side = int(float(min_side) * m);
    float cur_scale = 1.0f;
    const float factor = 0.709f;
    float scale;

    while (min_side >= 12) {
        scale = m * cur_scale;
        cur_scale = cur_scale * factor;
        min_side *= factor;

        int hs = int(std::ceil(height_ * scale));
        int ws = int(std::ceil(width_ * scale));
        stage_size_.push_back(cv::Size(ws, hs));
        Debug(("Create Pnet stage hs/ws " + std::to_string(hs) + "/" + std::to_string(ws)));
        pnet_stages_.push_back(std::make_shared<Pnet>(this->graph_, this->sess_, hs, ws, scale));
    }

    // Generate Rnet
    rnet_ = std::make_shared<Rnet>(this->graph_, this->sess_);

    // Generate Onet
    onet_ = std::make_shared<Onet>(this->graph_, this->sess_);

    return true;
}

bool FaceDetector::CreateSession() {
    std::shared_ptr<TF_SessionOptions> opt(TF_NewSessionOptions(),
                                           std::default_delete<TF_SessionOptions>());
    TF_Session *session = TF_NewSession(graph_.get(), opt.get(), status_.get());
    if (session) {
        sess_.reset(session, std::default_delete<TF_Session>());
    } else {
        if (TF_GetCode(status_.get()) != TF_OK) {
            std::cerr << "ERROR: Unable to create session" << std::string(TF_Message(status_.get()))
                      << std::endl;
        } else {
            std::cerr << "ERROR: Unable to create session" << std::endl;
        }
        return false;
    }

    return true;
}

bool FaceDetector::Init(std::string const filename) {
    bool ret = false;

    // load the graph
    ret = LoadGraph(filename);

    // create the session
    ret = CreateSession();

    // Create the MTCNN's architecture
    CreateArchitecture();
}

bool FaceDetector::LoadGraph(std::string const filename) {
    // Load graph.
    {
        std::shared_ptr<TF_Buffer> graph_def = ReadFile(filename.c_str());
        graph_.reset(TF_NewGraph(), std::default_delete<TF_Graph>());
        status_.reset(TF_NewStatus(), std::default_delete<TF_Status>());
        std::shared_ptr<TF_ImportGraphDefOptions> opts;
        opts.reset(TF_NewImportGraphDefOptions(), std::default_delete<TF_ImportGraphDefOptions>());
        TF_GraphImportGraphDef(graph_.get(), graph_def.get(), opts.get(), status_.get());

        if (TF_GetCode(status_.get()) != TF_OK) {
            std::cerr << "ERROR: Unable to import graph " << std::endl
                      << std::string(TF_Message(status_.get())) << std::endl;
            return false;
        }
    }
}

void FaceDetector::ProcessP(cv::Mat &fx3_image) {
    cv::Mat fx3_image_resized;
    total_pnet_boxes_.clear();
    for (int i = 0; i < stage_size_.size(); i++) {
        cv::resize(fx3_image, fx3_image_resized, stage_size_[i], 0, 0);
        pnet_stages_[i]->Process(fx3_image_resized);
        std::vector<FaceBox> istage_boxes = pnet_stages_[i].get()->final_candidate_boxes();
        total_pnet_boxes_.insert(total_pnet_boxes_.end(), istage_boxes.begin(), istage_boxes.end());
    }
}

void FaceDetector::Process(cv::Mat &u8x3_image) {
    cv::Mat fx3_image;

    float alpha = 0.0078125;
    float mean = 127.5;

    u8x3_image.convertTo(fx3_image, CV_32FC3);
    fx3_image = (fx3_image - mean) * alpha;
    fx3_image = fx3_image.t();

    std::vector<FaceBox> pnet_boxes;
    std::vector<FaceBox> rnet_boxes;
    std::vector<FaceBox> onet_boxes;
    ////////////////////////////////////
    //// PNET
    ProcessP(fx3_image);
    process_boxes(total_pnet_boxes_, height_, width_, pnet_boxes);
    Debug(" --- pnet_boxes " + std::to_string(pnet_boxes.size()));

    ///////////////////
    /// RNET
    rnet_->Process(fx3_image, pnet_boxes);
    total_rnet_boxes_ = rnet_->final_boxes();
    process_boxes(total_rnet_boxes_, height_, width_, rnet_boxes);
    Debug(" --- rnet_boxes " + std::to_string(rnet_boxes.size()));

    ///////////////////
    /// ONET
    onet_->Process(fx3_image, rnet_boxes);
    total_onet_boxes_ = onet_->final_boxes();

    Debug(" --- onet_boxes " + std::to_string(total_onet_boxes_.size()));
    for (unsigned int i = 0; i < total_onet_boxes_.size(); i++) {
        FaceBox &box = total_onet_boxes_[i];

        float h = box.x1 - box.x0 + 1;
        float w = box.y1 - box.y0 + 1;

        for (int j = 0; j < 5; j++) {
            box.landmark.x[j] = box.x0 + w * box.landmark.x[j] - 1;
            box.landmark.y[j] = box.y0 + h * box.landmark.y[j] - 1;
        }
    }

    // Get Final Result
    face_list_.clear();
    regress_boxes(total_onet_boxes_);
    nms_boxes(total_onet_boxes_, 0.7, NMSType::kNMS_MIN, face_list_);

    for (unsigned int i = 0; i < face_list_.size(); i++) {
        FaceBox &box = face_list_[i];

        std::swap(box.x0, box.y0);
        std::swap(box.x1, box.y1);

        for (int l = 0; l < 5; l++) {
            std::swap(box.landmark.x[l], box.landmark.y[l]);
        }
    }

#ifdef USEDEBUG

    for (auto f : pnet_boxes) {
        cv::Rect r(f.px0, f.py0, f.px1 - f.px0, f.py1 - f.py0);
        cv::rectangle(u8x3_image, r, cv::Scalar(255, 0, 0), 2);
    }

    for (auto f : rnet_boxes) {
        cv::Rect r(f.px0, f.py0, f.px1 - f.px0, f.py1 - f.py0);
        cv::rectangle(u8x3_image, r, cv::Scalar(0, 255, 0), 2);
    }

    for (auto f : face_list_) {
        for (int j = 0; j < 5; j++) {
            cv::circle(u8x3_image, cv::Point(f.landmark.x[j], f.landmark.y[j]), 10,
                       cv::Scalar(0, 0, 255), 5);
        }
    }

    cv::namedWindow("rects ", cv::WINDOW_NORMAL);
    cv::imshow("rects ", u8x3_image);
    cv::waitKey(-1);
#endif
}

static void free_buffer(void *data, std::size_t length) {
    (void)length;
    std::free(data);
}

std::shared_ptr<TF_Buffer> FaceDetector::ReadFile(std::string const filename) {
    std::shared_ptr<FILE> f(std::fopen(filename.c_str(), "rb"), std::fclose);
    if (!f) {
        std::cerr << "File " << std::string(filename) << " doesn't exist" << std::endl;
        return std::shared_ptr<TF_Buffer>();
    }

    std::fseek(f.get(), 0, SEEK_END);
    long fsize = ftell(f.get());
    std::fseek(f.get(), 0, SEEK_SET);

    std::shared_ptr<TF_Buffer> buf(TF_NewBuffer(), TF_DeleteBuffer);
    buf->data = ::malloc(fsize);
    std::fread(const_cast<void *>(buf->data), fsize, 1, f.get());
    buf->length = fsize;
    buf->data_deallocator = free_buffer;

    return buf;
}
