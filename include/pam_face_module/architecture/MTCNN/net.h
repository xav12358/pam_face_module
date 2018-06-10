#ifndef NET_H
#define NET_H

#include <tensorflow/c/c_api.h>
#include <memory>
#include <opencv2/opencv.hpp>

#include <pam_face_module/utils.h>


class Net
{
protected:
    bool is_init_;

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

public:
    Net(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session);

    virtual bool Init() = 0;
};

#endif // NET_H
