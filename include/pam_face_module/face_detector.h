#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <tensorflow/c/c_api.h>
#include <memory>

#include <pam_face_module/architecture/MTCNN/onet.h>
#include <pam_face_module/architecture/MTCNN/pnet.h>
#include <pam_face_module/architecture/MTCNN/rnet.h>

#include <pam_face_module/utils.h>

class FaceDetector {
    // Facedetector features
    bool is_init_;
    int height_, width_;
    int min_size_;

    // Load graph.
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Status> status_;

    // Create session.
    std::shared_ptr<TF_Session> sess_;

    // Architecture
    std::vector<std::shared_ptr<Pnet>> pnet_stages_;
    std::shared_ptr<Rnet> rnet_;
    std::shared_ptr<Onet> onet_;

public:

    FaceDetector(int h, int w, int min_size, const std::string finename);
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

    ////////////////
    /// \brief Face_detector::init
    /// \param fileName
    ///
    bool Init(const std::string filename);

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

#endif  // FACE_DETECTOR_H
