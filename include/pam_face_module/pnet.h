#ifndef PNET_H
#define PNET_H

#include <memory>
#include <tensorflow/c/c_api.h>

class Pnet {

    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;
    std::shared_ptr<TF_ImportGraphDefOptions> opts_;

public:
  Pnet(std::shared_ptr<TF_Graph> graph);
};

#endif // PNET_H
