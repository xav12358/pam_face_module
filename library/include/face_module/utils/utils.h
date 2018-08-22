#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <memory>
#include <tensorflow/c/c_api.h>
#include <vector>

#include <opencv2/opencv.hpp>

// Debug
#define USEDEBUG

#ifdef USEDEBUG
#define Debug(x) std::cout << x << std::endl;
#else
#define Debug(x)
#endif

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

static void nullDeallocator(void *ptr, std::size_t len, void *arg) {
  (void)len;
  (void)arg;
  (void)ptr;
}

struct FaceLandmark {
  float x[5];
  float y[5];
};

struct FaceBox {
  float x0;
  float y0;
  float x1;
  float y1;

  /* confidence score */
  float score;

  /*regression scale */

  float regress[4];

  /* padding stuff*/
  float px0;
  float py0;
  float px1;
  float py1;

  FaceLandmark landmark;
};

typedef enum { kNMS_UNION, kNMS_MIN } NMSType;

void copy_one_patch(const cv::Mat &img, FaceBox &input_box, float *data_to,
                    int height, int width);
void copy_one_patch_inv(const cv::Mat &img, FaceBox &input_box, float *data_to,
                                                               int height, int width);
void copy_one_image(const cv::Mat &img, float *data_to);
void dummy_deallocator(void *data, size_t len, void *arg);
void nms_boxes(std::vector<FaceBox> &input, float threshold, int type,
               std::vector<FaceBox> &output);
void regress_boxes(std::vector<FaceBox> &rects);
void square_boxes(std::vector<FaceBox> &rects);
void padding(int img_h, int img_w, std::vector<FaceBox> &rects);
void process_boxes(std::vector<FaceBox> &input, int img_h, int img_w,
                   std::vector<FaceBox> &rects);

std::shared_ptr<TF_Buffer> ReadFile(std::string const filename);

#endif // UTILS_H
