#ifndef ONET_H
#define ONET_H

#include <tensorflow/c/c_api.h>
#include <memory>
#include <vector>

#include <pam_face_module/architecture/MTCNN/net.h>
#include <pam_face_module/utils.h>

class Onet :public Net{

    ///
    const int kHeight_ = 48;
    const int kWidth_ = 48;
    const float kThreshold = 0.9;

    // Prepare inputs.
    TF_Output run_inputs_[1];
    TF_Tensor *run_inputs_tensors_[1];

    // Generate output
    TF_Output run_outputs_[3];
    TF_Tensor *run_output_tensors_[3];


    //Candidate
    std::vector<FaceBox> final_boxes_;

 public:
    ////////////////
    /// \brief Onet
    /// \param graph
    /// \param session
    ///
    Onet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

    ~Onet(){};



    ///////////////
    /// \brief final_boxes
    /// \return
    ///
    std::vector<FaceBox> final_boxes() const;

    ///////////////////
    /// \brief Init
    /// \return
    ///
    virtual bool Init();

    ///////////////
    /// \brief Process
    /// \param img
    /// \param pnet_candidates
    ///
    void Process(cv::Mat &img, std::vector<FaceBox> & rnet_candidates);
};

#endif  // ONET_H
