#ifndef RNET_H
#define RNET_H

#include <memory>
#include <tensorflow/c/c_api.h>

class Rnet {

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;


public:
  Rnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);
};

#endif // RNET_H

