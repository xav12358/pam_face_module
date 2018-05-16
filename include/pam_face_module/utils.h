#ifndef UTILS_H
#define UTILS_H

#include <memory>
#include <iostream>
#include <tensorflow/c/c_api.h>


namespace std {

template <>
struct default_delete<TF_Session> {
    void operator()(TF_Session *session) const {
        if (session) {
            std::shared_ptr<TF_Status> status(TF_NewStatus(), TF_DeleteStatus);
            TF_DeleteSession(session, status.get());
            if (TF_GetCode(status.get()) != TF_OK) {
                std::cerr << "Error while deleting session: " << TF_Message(status.get())
                          << std::endl;
            }
        }
    }
};

#define SET_TF_DEFAULT_DELETER(__class)             \
    template <>                                     \
    struct default_delete<TF_##__class> {           \
        void operator()(TF_##__class *elem) const { \
            if (elem) {                             \
                TF_Delete##__class(elem);           \
            }                                       \
        }                                           \
    }

SET_TF_DEFAULT_DELETER(Graph);
SET_TF_DEFAULT_DELETER(Status);
SET_TF_DEFAULT_DELETER(Tensor);
SET_TF_DEFAULT_DELETER(ImportGraphDefOptions);
SET_TF_DEFAULT_DELETER(SessionOptions);

}  // namespace std


#endif // UTILS_H
