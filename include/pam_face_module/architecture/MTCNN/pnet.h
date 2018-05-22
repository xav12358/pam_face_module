#ifndef PNET_H
#define PNET_H

#include <tensorflow/c/c_api.h>
#include <memory>
#include <opencv2/opencv.hpp>

#include <pam_face_module/utils.h>

class Pnet {
    bool is_init_;

    // Current stage feature
    int height_, width_;
    float scale_;
    const float kPnetThreshold = 0.6f;

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

//     Gerenate input.
    std::shared_ptr<float> raw_input_data_;
    std::shared_ptr<int64_t> raw_input_dims_;


    // Prepare inputs.
    TF_Output run_inputs_[1];
    std::shared_ptr<TF_Tensor> input_tensor_;
    TF_Tensor *run_inputs_tensors_[1];

    // Generate output
    TF_Output run_outputs_[2];
    TF_Tensor *run_output_tensors_[2];

    //Candidate
    std::vector<FaceBox> final_candidate_boxes_;


 public:
    /////////////////
    /// \brief Pnet
    /// \param graph
    /// \param session
    ///
    Pnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session, int h, int w, float scale);

    ////////////////
    ~Pnet() {}

    ///////////////
    /// \brief Process
    /// \param u8x3_image
    ///
    void Process(const cv::Mat &fx3_image);

    //////////////////////
    /// \brief final_candidate_boxes
    /// \return
    ///
    std::vector<FaceBox> final_candidate_boxes() const;

private:
    void FeedML(const cv::Mat &u8x3_image);
    void FetchData();
    bool Init();
    void GenerateBoundingBox(const float *confidence_data, int confidence_size,
                             const float *reg_data, float scale, float threshold, int feature_h,
                             int feature_w, std::vector<FaceBox> &output, bool transposed);

    void RunSession();
};

#endif  // PNET_H
