#ifndef PNET_H
#define PNET_H

#include <memory>
#include <tensorflow/c/c_api.h>

class Pnet {

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;


public:
  Pnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

private:
  bool init();
};

#endif // PNET_H
