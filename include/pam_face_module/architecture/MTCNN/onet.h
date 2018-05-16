#ifndef ONET_H
#define ONET_H

#include <tensorflow/c/c_api.h>
#include <memory>

class Onet {
    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

 public:
    ////////////////
    /// \brief Onet
    /// \param graph
    /// \param session
    ///
    Onet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

    ~Onet(){};
};

#endif  // ONET_H
