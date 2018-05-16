#include "pam_face_module/architecture/MTCNN/rnet.h"

Rnet::Rnet(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) {
    graph_ = graph;
    sess_ = session;



}


