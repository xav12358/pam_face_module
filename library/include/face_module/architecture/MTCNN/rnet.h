#ifndef RNET_H
#define RNET_H

#include <tensorflow/c/c_api.h>
#include <memory>

#include <face_module/architecture/MTCNN/net.h>


class Rnet : public Net{

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


    //Candidate
    std::vector<FaceBox> final_boxes_;

 public:
    //////////////////
    /// \brief Rnet
    /// \param graph
    /// \param session
    ///
    Rnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

    ////////////////
    ~Rnet() {}

    ////////////////////
    /// \brief Init
    ///
    bool Init();

    ///////////////
    /// \brief Process
    /// \param img
    /// \param pnet_candidates
    ///
    void Process(cv::Mat &img, std::vector<FaceBox> & pnet_candidates);


    //////////////////////
    /// \brief final_boxes
    /// \return
    ///
    std::vector<FaceBox> final_boxes() const;
};

#endif  // RNET_H
