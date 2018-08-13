#include "face_module/architecture/MTCNN/net.h"

Net::Net(std::shared_ptr<TF_Graph> graph, std::shared_ptr<TF_Session> session) {
    graph_ = graph;
    sess_ = session;

    status_.reset(TF_NewStatus(), std::default_delete<TF_Status>());
}


Net::Net() {
    status_.reset(TF_NewStatus(), std::default_delete<TF_Status>());
}
