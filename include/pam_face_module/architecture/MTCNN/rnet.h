#ifndef RNET_H
#define RNET_H

#include <tensorflow/c/c_api.h>
#include <memory>
#include <pam_face_module/utils.h>

class Rnet {
    bool is_init_;

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

    ///
    const int kHeight_ = 24;
    const int kWidth_ = 24;
    const float kThreshold = 0.7;

    // Prepare inputs.
    TF_Output run_inputs_[1];
    TF_Tensor *run_inputs_tensors_[1];

    // Generate output
    TF_Output run_outputs_[2];
    TF_Tensor *run_output_tensors_[2];

 public:
    //////////////////
    /// \brief Rnet
    /// \param graph
    /// \param session
    ///
    Rnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

    ////////////////
    ~Rnet() {}


    void Process(cv::Mat &img, std::vector<FaceBox> & pnet_candidates);

private:
    ////////////////////
    /// \brief Init
    ///
    bool Init();
};

#endif  // RNET_H
