#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <memory>
#include <tensorflow/c/c_api.h>

#include <face_module/architecture/MTCNN/onet.h>
#include <face_module/architecture/MTCNN/pnet.h>
#include <face_module/architecture/MTCNN/rnet.h>

class Face_detector_test;
class FaceDetector {

    friend class Face_detector_test;

    // Facedetector features
    bool is_init_;
    int height_, width_;
    int min_size_;
    std::vector<cv::Size> stage_size_;

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

    // Architecture
    std::vector<std::shared_ptr<Pnet>> pnet_stages_;
    std::shared_ptr<Rnet> rnet_;
    std::shared_ptr<Onet> onet_;

    // Initial box position for each part
    std::vector<FaceBox> total_pnet_boxes_;
    std::vector<FaceBox> total_rnet_boxes_;
    std::vector<FaceBox> total_onet_boxes_;
    std::vector<FaceBox> face_list_;

public:
  FaceDetector(int h, int w, int min_size);

  ////////////////
  /// \brief Face_detector::init
  /// \param fileName
  ///
  bool Init(const std::string filename);

  //////////////////
  /// \brief Process
  /// \param u8x3_image
  ///
  void Process(cv::Mat u8x3_image);

  std::vector<FaceBox> total_pnet_boxes() const;

  std::vector<FaceBox> total_rnet_boxes() const;

  std::vector<FaceBox> total_onet_boxes() const;

  std::vector<FaceBox> face_list() const;

private:

  //////////////////////
  /// \brief CreateArchitecture
  /// \return
  ///
  bool CreateArchitecture();

  /////////////
  /// \brief createSession
  /// \return
  ///
  bool CreateSession();


  ////////////////////
  /// \brief Face_detector::loadGraph
  /// \param fileName
  ///
  bool LoadGraph(std::string const fileName);


  //////////////
  /// \brief ProcessP
  ///
  void ProcessP(cv::Mat &fx3_image);

};

#endif // FACE_DETECTOR_H
