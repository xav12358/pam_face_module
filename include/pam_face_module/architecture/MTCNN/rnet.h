#ifndef RNET_H
#define RNET_H

#include <tensorflow/c/c_api.h>
#include <memory>

class Rnet {
    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

 public:
    //////////////////
    /// \brief Rnet
    /// \param graph
    /// \param session
    ///
    Rnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

    ////////////////
    ~Rnet() {}
};

#endif  // RNET_H
