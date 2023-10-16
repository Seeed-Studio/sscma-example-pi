// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// modified 1-14-2023 Q-engineering

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <iostream>
#include <options.hpp>
#include <net.h>

using namespace std::chrono;

const char *class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class YOLOV8
{
public:
    YOLOV8();
    int load_param(const char *path);
    int load_model(const char *path);
    int init(int num_threads, std::vector<int> &input_shape, std::vector<float> &input_mean, std::vector<float> &input_std);
    int detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold = 0.25f, float nms_threshold = 0.45f);
    int draw(cv::Mat &rgb, const std::vector<Object> &objects, std::vector<std::string> &classes);

private:
    ncnn::Net net;
    int input[4];
    float mean_vals[3];
    float norm_vals[3];
};

static float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &pred, float prob_threshold, std::vector<Object> &objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 80;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float *scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void *)pred.row(i));
            {
                ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 4;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float *dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob * 100;

            objects.push_back(obj);
        }
    }
}

YOLOV8::YOLOV8()
{
    net.clear();
    mean_vals[0] = 0;
    mean_vals[1] = 0;
    mean_vals[2] = 0;
    norm_vals[0] = 1 / 255.f;
    norm_vals[1] = 1 / 255.f;
    norm_vals[2] = 1 / 255.f;
    input[0] = 1;
    input[1] = 3;
    input[2] = 640;
    input[3] = 640;
}

int YOLOV8::load_param(const char *path)
{
    return net.load_param(path);
}

int YOLOV8::load_model(const char *path)
{
    return net.load_model(path);
}

int YOLOV8::init(int num_threads, std::vector<int> &input_shape, std::vector<float> &input_mean, std::vector<float> &input_std)
{

    net.opt = ncnn::Option();

    net.opt.num_threads = num_threads;

    input[0] = input_shape[0];
    input[1] = input_shape[1];
    input[2] = input_shape[2];
    input[3] = input_shape[3];

    mean_vals[0] = input_mean[0];
    mean_vals[1] = input_mean[1];
    mean_vals[2] = input_mean[2];

    norm_vals[0] = input_std[0];
    norm_vals[1] = input_std[1];
    norm_vals[2] = input_std[2];

    return 0;
}

int YOLOV8::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)input[2] / w;
        w = input[2];
        h = h * scale;
    }
    else
    {
        scale = (float)input[3] / h;
        h = input[3];
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle

    int wpad = input[2] - w;
    int hpad = input[3] - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    ncnn::Mat out;

    ex.extract("output", out);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object &a, const Object &b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLOV8::draw(cv::Mat &rgb, const std::vector<Object> &objects, std::vector<std::string> &classes)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];

        cv::rectangle(rgb, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        if (obj.label <= classes.size())
            sprintf(text, "%s %.1f%%", classes[obj.label].c_str(), obj.prob);
        else
            sprintf(text, "%d %.1f%%", obj.label, obj.prob);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

int main(int argc, char **argv)
{

    std::vector<std::string> classes;
    for (int i = 0; i < sizeof(class_names) / sizeof(class_names[0]); i++)
    {
        classes.push_back(class_names[i]);
    }
    Options options("../models/sscma-yolov8/model.param", "../models/sscma-yolov8/model.bin", classes);

    options.Parser(argc, argv);

    YOLOV8 yolov8;
    yolov8.init(options.thread_num, options.input_shape, options.input_mean, options.input_std);
    yolov8.load_param(options.model_path.c_str());
    yolov8.load_model(options.weights_path.c_str());

    if (options.input_type == Options::Image)
    {
        cv::Mat m = cv::imread(options.input_path, 1);
        std::vector<Object> objects;
        auto start = high_resolution_clock::now();
        yolov8.detect(m, objects, options.score_threshold, options.iou_threshold);
        auto stop = high_resolution_clock::now();
        yolov8.draw(m, objects, options.classes);
        auto duration = duration_cast<milliseconds>(stop - start);
        float fps = 1000.0 / duration.count();
        std::cout << "Estimated frames per second : " << fps << ":" << duration.count() << std::endl;
        std::cout << "Objects: " << objects.size() << std::endl;
        for (auto obj : objects)
        {
            std::cout << "\t" << options.classes[obj.label] << ":" << obj.prob << " [" << obj.rect.x << ", " << obj.rect.y << ", " << obj.rect.width << ", " << obj.rect.height << "]" << std::endl;
        }
        if (options.save_result)
            cv::imwrite(options.output_path, m);
        if (options.show_result)
            cv::imshow("sscma yolov8", m);
    }

    if (options.input_type == Options::Video)
    {
        cv::VideoCapture cap;
        cv::Mat m;
        if (!cap.open(options.input_path))
        {
            std::cerr << "\x1b[31m"
                      << "ERROR: Can't open video stream " << options.input_path
                      << "\x1b[0m" << std::endl;
            return 0;
        }

        double rate = cap.get(cv::CAP_PROP_FPS);
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        // std::cout << "width: " << width << std::endl;
        // std::cout << "height: " << height << std::endl;
        // std::cout << "rate: " << rate << std::endl;
        cv::VideoWriter writer(options.output_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), rate, cv::Size(width, height), true);
        for (;;)
        {
            cap >> m;
            if (m.empty())
                break; // end of video stream
            auto start = high_resolution_clock::now();
            std::vector<Object> objects;
            yolov8.detect(m, objects, options.score_threshold, options.iou_threshold);
            auto stop = high_resolution_clock::now();
            yolov8.draw(m, objects, options.classes);
            auto duration = duration_cast<milliseconds>(stop - start);
            float fps = 1000.0 / duration.count();
            std::cout << "Estimated frames per second : " << fps << ":" << duration.count() << std::endl;
            std::cout << "Objects: " << objects.size() << std::endl;
            for (auto obj : objects)
            {
                std::cout << "\t" << options.classes[obj.label] << ":" << obj.prob << " [" << obj.rect.x << ", " << obj.rect.y << ", " << obj.rect.width << ", " << obj.rect.height << "]" << std::endl;
            }
            if (options.save_result)
            {
                writer.write(m);
            }

            if (options.show_result)
            {
                cv::imshow("sscma yolov8", m);
                if (cv::waitKey(1) == 27)
                    break; // stop capturing by pressing ESC
            }
        }
    }

    if (options.input_type == Options::Camera)
    {

        cv::VideoCapture cap;
        cv::Mat m;
        if (!cap.open(std::stoi(options.input_path)))
        {
            std::cerr << "\x1b[31m"
                      << "ERROR: Can't open Camera " << options.input_path
                      << "\x1b[0m" << std::endl;
            return 0;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cv::VideoWriter writer(options.output_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 25, cv::Size(640, 480), true);

        for (;;)
        {
            cap >> m;
            if (m.empty())
                break; // end of video stream
            auto start = high_resolution_clock::now();
            std::vector<Object> objects;
            yolov8.detect(m, objects, options.score_threshold, options.iou_threshold);
            auto stop = high_resolution_clock::now();
            yolov8.draw(m, objects, options.classes);
            auto duration = duration_cast<milliseconds>(stop - start);
            float fps = 1000.0 / duration.count();
            std::cout << "Estimated frames per second : " << fps << ":" << duration.count() << std::endl;
            std::cout << "Objects: " << objects.size() << std::endl;
            for (auto obj : objects)
            {
                std::cout << "\t" << options.classes[obj.label] << ":" << obj.prob << " [" << obj.rect.x << ", " << obj.rect.y << ", " << obj.rect.width << ", " << obj.rect.height << "]" << std::endl;
            }
            if (options.save_result)
            {
                writer.write(m);
            }

            if (options.show_result)
            {
                cv::imshow("sscma yolov8", m);
                if (cv::waitKey(1) == 27)
                    break; // stop capturing by pressing ESC
            }
        }
    }

    if (options.save_result)
    {
        std::cout << "Save result to " << options.output_path << std::endl;
    }

    return 0;
}