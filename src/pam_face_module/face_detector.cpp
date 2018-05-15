#include "pam_face_module/face_detector.h"

#include <iostream>

namespace std {

template <> struct default_delete<TF_Session> {
  void operator()(TF_Session *session) const {
    if (session) {
      std::shared_ptr<TF_Status> status(TF_NewStatus(), TF_DeleteStatus);
      TF_DeleteSession(session, status.get());
      if (TF_GetCode(status.get()) != TF_OK) {
        std::cerr << "Error while deleting session: "
                  << TF_Message(status.get()) << std::endl;
      }
    }
  }
};

#define SET_TF_DEFAULT_DELETER(__class)                                        \
  template <> struct default_delete<TF_##__class> {                            \
    void operator()(TF_##__class *elem) const {                                \
      if (elem) {                                                              \
        TF_Delete##__class(elem);                                              \
      }                                                                        \
    }                                                                          \
  }

SET_TF_DEFAULT_DELETER(Graph);
SET_TF_DEFAULT_DELETER(Status);
SET_TF_DEFAULT_DELETER(Tensor);
SET_TF_DEFAULT_DELETER(ImportGraphDefOptions);
SET_TF_DEFAULT_DELETER(SessionOptions);

} // namespace std

Face_detector::Face_detector() {}

bool Face_detector::createSession() {

  std::shared_ptr<TF_SessionOptions> opt(
      TF_NewSessionOptions(), std::default_delete<TF_SessionOptions>());
  TF_Session *session = TF_NewSession(graph_.get(), opt.get(), status_.get());
  if (session) {
    sess_.reset(session, std::default_delete<TF_Session>());
  } else {
    if (TF_GetCode(status_.get()) != TF_OK) {
      std::fprintf(stderr, "ERROR: Unable to create session %s\n",
                   TF_Message(status_.get()));
    } else {
      std::fprintf(stderr, "ERROR: Unable to create session\n");
    }
    return false;
  }
}

void Face_detector::init(std::string const fileName) {

  // load the graph
  loadGraph(fileName);

  // create the session
  createSession();
}

bool Face_detector::loadGraph(std::string const fileName) {
  // Load graph.
  {
    std::shared_ptr<TF_Buffer> graph_def = read_file(fileName.c_str());
    graph_.reset(TF_NewGraph(), std::default_delete<TF_Graph>());
    status_.reset(TF_NewStatus(), std::default_delete<TF_Status>());
    std::shared_ptr<TF_ImportGraphDefOptions> opts;
    opts.reset(TF_NewImportGraphDefOptions(),
               std::default_delete<TF_ImportGraphDefOptions>());
    TF_GraphImportGraphDef(graph_.get(), graph_def.get(), opts.get(),
                           status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
      std::fprintf(stderr, "ERROR: Unable to import graph %s\n",
                   TF_Message(status_.get()));
      return false;
    }
  }
}

static void free_buffer(void *data, std::size_t length) {
  (void)length;
  std::free(data);
}

std::shared_ptr<TF_Buffer>
Face_detector::read_file(std::string const fileName) {
  std::shared_ptr<FILE> f(std::fopen(fileName.c_str(), "rb"), std::fclose);

  if (!f) {
    std::cerr << "File " << std::string(fileName) << " doesn't exist"
              << std::endl;
    return std::shared_ptr<TF_Buffer>();
  }

  std::fseek(f.get(), 0, SEEK_END);
  long fsize = ftell(f.get());
  std::fseek(f.get(), 0, SEEK_SET);

  std::shared_ptr<TF_Buffer> buf(TF_NewBuffer(), TF_DeleteBuffer);
  buf->data = ::malloc(fsize);
  std::fread(const_cast<void *>(buf->data), fsize, 1, f.get());
  buf->length = fsize;
  buf->data_deallocator = free_buffer;

  return buf;
}
