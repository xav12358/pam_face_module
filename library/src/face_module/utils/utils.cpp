#include <face_module/utils/utils.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>


static void free_buffer(void *data, std::size_t length) {
    (void)length;
    std::free(data);
}

void copy_one_patch(const cv::Mat &img, FaceBox &input_box, float *data_to,
                    int height, int width) {
  cv::Mat resized(height, width, CV_32FC3, data_to);

  cv::Mat chop_img = img(cv::Range(input_box.py0, input_box.py1),
                         cv::Range(input_box.px0, input_box.px1));

  int pad_top = std::abs(input_box.py0 - input_box.y0);
  int pad_left = std::abs(input_box.px0 - input_box.x0);
  int pad_bottom = std::abs(input_box.py1 - input_box.y1);
  int pad_right = std::abs(input_box.px1 - input_box.x1);

  cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left,
                     pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

  cv::resize(chop_img, resized, cv::Size(width, height), 0, 0);
}

void copy_one_patch_inv(const cv::Mat &img, FaceBox &input_box, float *data_to,
                    int height, int width) {
    std::cout << "copy_one_patch_inv " << std::endl;
  cv::Mat resized(height, width, CV_32FC3, data_to);
  std::cout << "copy_one_patch_inv " << std::endl;
    std::cout << "  " << input_box.px0 << " / " <<input_box.px1
              << "  " << input_box.py0<< " / " << input_box.py1 <<std::endl;
  cv::Mat chop_img = img(cv::Range(input_box.px0, input_box.px1),
                         cv::Range(input_box.py0, input_box.py1));
  cv::imshow("copy_one_patch_inv",chop_img);
  cv::waitKey(-1);

  int pad_left = std::abs(input_box.py0 - input_box.y0);
  int pad_top  = std::abs(input_box.px0 - input_box.x0);
  int pad_right = std::abs(input_box.py1 - input_box.y1);
  int pad_bottom  = std::abs(input_box.px1 - input_box.x1);

  cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left,
                     pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

  cv::resize(chop_img, resized, cv::Size(width, height), 0, 0);
}

void copy_one_image(const cv::Mat &img, float *data_to) {
  cv::Mat resized(img.rows, img.cols, CV_32FC3, data_to);
  cv::resize(img, resized, cv::Size(img.cols, img.rows), 0, 0);
}

void dummy_deallocator(void *data, size_t len, void *arg) {}

void nms_boxes(std::vector<FaceBox> &input, float threshold, int type_NMS,
               std::vector<FaceBox> &output) {
  std::sort(input.begin(), input.end(), [](const FaceBox &a, const FaceBox &b) {
    return a.score > b.score;
  });

  size_t box_num = input.size();

  std::vector<int> merged(box_num, 0);

  for (size_t i = 0; i < box_num; i++) {
    if (merged[i])
      continue;

    output.push_back(input[i]);

    float h0 = input[i].y1 - input[i].y0 + 1;
    float w0 = input[i].x1 - input[i].x0 + 1;

    float area0 = h0 * w0;

    for (size_t j = i + 1; j < box_num; j++) {
      if (merged[j])
        continue;

      float inner_x0 = std::max(input[i].x0, input[j].x0);
      float inner_y0 = std::max(input[i].y0, input[j].y0);

      float inner_x1 = std::min(input[i].x1, input[j].x1);
      float inner_y1 = std::min(input[i].y1, input[j].y1);

      float inner_h = inner_y1 - inner_y0 + 1;
      float inner_w = inner_x1 - inner_x0 + 1;

      if (inner_h <= 0 || inner_w <= 0)
        continue;

      float inner_area = inner_h * inner_w;

      float h1 = input[j].y1 - input[j].y0 + 1;
      float w1 = input[j].x1 - input[j].x0 + 1;

      float area1 = h1 * w1;

      float score;

      switch (type_NMS) {
      case NMSType::kNMS_MIN:
        score = inner_area / std::min(area0, area1);
        break;

      case NMSType::kNMS_UNION:
        score = inner_area / (area0 + area1 - inner_area);
        break;
      }

      if (score > threshold)
        merged[j] = 1;
    }
  }
}

void regress_boxes(std::vector<FaceBox> &rects) {
  for (unsigned int i = 0; i < rects.size(); i++) {
    FaceBox &box = rects[i];

    float h = box.y1 - box.y0 + 1;
    float w = box.x1 - box.x0 + 1;

    box.x0 = box.x0 + w * box.regress[0];
    box.y0 = box.y0 + h * box.regress[1];
    box.x1 = box.x1 + w * box.regress[2];
    box.y1 = box.y1 + h * box.regress[3];
  }
}

void square_boxes(std::vector<FaceBox> &rects) {

  for (unsigned int i = 0; i < rects.size(); i++) {
    float h = rects[i].y1 - rects[i].y0 + 1;
    float w = rects[i].x1 - rects[i].x0 + 1;

    float l = std::max(h, w);

    rects[i].x0 = rects[i].x0 + (w - l) * 0.5;
    rects[i].y0 = rects[i].y0 + (h - l) * 0.5;
    rects[i].x1 = rects[i].x0 + l - 1;
    rects[i].y1 = rects[i].y0 + l - 1;
  }
}

void padding(int img_h, int img_w, std::vector<FaceBox> &rects) {
  for (unsigned int i = 0; i < rects.size(); i++) {
    rects[i].px0 = std::max(rects[i].x0, 1.0f);
    rects[i].py0 = std::max(rects[i].y0, 1.0f);
    rects[i].px1 = std::min(rects[i].x1, (float)img_w);
    rects[i].py1 = std::min(rects[i].y1, (float)img_h);
  }
}

void process_boxes(std::vector<FaceBox> &input, int img_h, int img_w,
                   std::vector<FaceBox> &rects) {
  nms_boxes(input, 0.7, NMSType::kNMS_UNION, rects);
  regress_boxes(rects);
  square_boxes(rects);
  padding(img_h, img_w, rects);
}


std::shared_ptr<TF_Buffer> ReadFile(std::string const filename) {
    std::shared_ptr<FILE> f(std::fopen(filename.c_str(), "rb"), std::fclose);
    if (!f) {
        std::cerr << "File " << std::string(filename) << " doesn't exist" << std::endl;
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
