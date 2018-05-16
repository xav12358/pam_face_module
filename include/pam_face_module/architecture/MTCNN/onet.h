#ifndef ONET_H
#define ONET_H

#include <memory>
#include <tensorflow/c/c_api.h>

class Onet {

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;


public:
  Onet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);
};

#endif // ONET_H
