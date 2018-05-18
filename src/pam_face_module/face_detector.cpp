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


        std::cout << "Create Pnet stage hs/ws " << hs <<"/" << ws << std::endl;
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
            std::fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status_.get()));
        } else {
            std::fprintf(stderr, "ERROR: Unable to create session\n");
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
            std::fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status_.get()));
            return false;
        }
    }
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
