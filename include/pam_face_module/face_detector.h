#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <memory>
#include <tensorflow/c/c_api.h>

#include <pam_face_module/architecture/MTCNN/pnet.h>
#include <pam_face_module/architecture/MTCNN/rnet.h>
#include <pam_face_module/architecture/MTCNN/onet.h>

#include <pam_face_module/utils.h>

class FaceDetector {

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

    // Architecture
    std::shared_ptr<Pnet> Pnet_;
    std::shared_ptr<Rnet> Rnet_;
    std::shared_ptr<Onet> Onet_;


public:
  FaceDetector();

private:

  /////////////
  /// \brief createSession
  /// \return
  ///
  bool CreateSession();

  ////////////////
  /// \brief Face_detector::init
  /// \param fileName
  ///
  void Init(std::string const fileName);


  ////////////////////
  /// \brief Face_detector::loadGraph
  /// \param fileName
  ///
  bool LoadGraph(std::string const fileName);

  /////////////////
  /// \brief Face_detector::read_file
  /// \param fileName
  /// \return
  ///
  std::shared_ptr<TF_Buffer> ReadFile(std::string const fileName);
};

#endif // FACE_DETECTOR_H
