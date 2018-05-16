#ifndef PNET_H
#define PNET_H

#include <tensorflow/c/c_api.h>
#include <memory>

#include <pam_face_module/utils.h>

class Pnet {

    bool is_init_;

    // Current stage feature
    int height_, width_;

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

 public:
    /////////////////
    /// \brief Pnet
    /// \param graph
    /// \param session
    ///
    Pnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session, int h, int w);

    ////////////////
    ~Pnet() {}

 private:
    bool Init();
};

#endif  // PNET_H
