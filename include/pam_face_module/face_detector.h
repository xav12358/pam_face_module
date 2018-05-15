#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <memory>
#include <tensorflow/c/c_api.h>

class Face_detector {

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

public:
  Face_detector();

private:

  /////////////
  /// \brief createSession
  /// \return
  ///
  bool createSession();

  ////////////////
  /// \brief Face_detector::init
  /// \param fileName
  ///
  void init(std::string const fileName);


  ////////////////////
  /// \brief Face_detector::loadGraph
  /// \param fileName
  ///
  bool loadGraph(std::string const fileName);

  /////////////////
  /// \brief Face_detector::read_file
  /// \param fileName
  /// \return
  ///
  std::shared_ptr<TF_Buffer> read_file(std::string const fileName);
};

#endif // FACE_DETECTOR_H
