#include <pam_face_module/utils.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>

void nms_boxes(std::vector<FaceBox> &input, float threshold, int type_NMS,
               std::vector<FaceBox> &output) {
    std::sort(input.begin(), input.end(),
              [](const FaceBox &a, const FaceBox &b) { return a.score > b.score; });

    size_t box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (size_t i = 0; i < box_num; i++) {
        if (merged[i]) continue;

        output.push_back(input[i]);

        float h0 = input[i].y1 - input[i].y0 + 1;
        float w0 = input[i].x1 - input[i].x0 + 1;

        float area0 = h0 * w0;

        for (size_t j = i + 1; j < box_num; j++) {
            if (merged[j]) continue;

            float inner_x0 = std::max(input[i].x0, input[j].x0);
            float inner_y0 = std::max(input[i].y0, input[j].y0);

            float inner_x1 = std::min(input[i].x1, input[j].x1);
            float inner_y1 = std::min(input[i].y1, input[j].y1);

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0) continue;

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

            if (score > threshold) merged[j] = 1;
        }
    }
}
