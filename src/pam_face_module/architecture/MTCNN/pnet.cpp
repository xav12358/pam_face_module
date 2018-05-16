#include "pam_face_module/architecture/MTCNN/pnet.h"

Pnet::Pnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) {
    graph_ = graph;
    sess_ = session;
}

bool Pnet::Init() {


    return true;
}
