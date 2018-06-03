#include "pam_face_module/architecture/MTCNN/net.h"

Net::Net(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) {
    graph_ = graph;
    sess_ = session;
}