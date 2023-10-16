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

class YOLOV5
{
public:
    YOLOV5();
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

static void generate_proposals(const ncnn::Mat &pred, float prob_threshold, std::vector<Object> &objects)
{
    auto num_record{pred.h};
    auto num_element{pred.w};
    auto num_class{static_cast<uint8_t>(num_element - 5)};

    for (int i = 0; i < num_record; i++)
    {
        auto idx{i * num_element};
        auto score{(pred[idx + 4])};

        if (score > prob_threshold * 100)
        {
            Object obj;
            obj.prob = score;

            // find label with max score
            float max = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float confidence = pred[idx + 5 + k];
                if (confidence > max)
                {
                    max = confidence;
                    obj.label = k;
                }
            }

            obj.rect.width = pred[idx + 2];
            obj.rect.height = pred[idx + 3];
            obj.rect.x = pred[idx] - obj.rect.width / 2.f;
            obj.rect.y = pred[idx + 1] - obj.rect.height / 2.f;

            objects.push_back(obj);
        }
    }
}

YOLOV5::YOLOV5()
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

int YOLOV5::load_param(const char *path)
{
    return net.load_param(path);
}

int YOLOV5::load_model(const char *path)
{
    return net.load_model(path);
}

int YOLOV5::init(int num_threads, std::vector<int> &input_shape, std::vector<float> &input_mean, std::vector<float> &input_std)
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

int YOLOV5::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold, float nms_threshold)
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

    ex.input("in0", in_pad);

    std::vector<Object> proposals;

    ncnn::Mat out;

    ex.extract("out0", out);

    generate_proposals(out, prob_threshold, proposals);
    //  sort all proposals by score from highest to lowest
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

int YOLOV5::draw(cv::Mat &rgb, const std::vector<Object> &objects, std::vector<std::string> &classes)
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
    Options options("../models/sscma-yolov5/model.param", "../models/sscma-yolov5/model.bin", classes);

    options.Parser(argc, argv);

    YOLOV5 yolov5;
    yolov5.init(options.thread_num, options.input_shape, options.input_mean, options.input_std);
    yolov5.load_param(options.model_path.c_str());
    yolov5.load_model(options.weights_path.c_str());

    if (options.input_type == Options::Image)
    {
        cv::Mat m = cv::imread(options.input_path, 1);
        std::vector<Object> objects;
        auto start = high_resolution_clock::now();
        yolov5.detect(m, objects, options.score_threshold, options.iou_threshold);
        auto stop = high_resolution_clock::now();
        yolov5.draw(m, objects, options.classes);
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
            cv::imshow("sscma yolov5", m);
    }

    if (options.input_type == Options::Video)
    {
        cv::VideoCapture cap;
        cv::Mat m;
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
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
        cv::VideoWriter writer(options.output_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), rate, cv::Size(width, height), true);
        for (;;)
        {
            cap >> m;
            if (m.empty())
                break; // end of video stream
            auto start = high_resolution_clock::now();
            std::vector<Object> objects;
            yolov5.detect(m, objects, options.score_threshold, options.iou_threshold);
            auto stop = high_resolution_clock::now();
            yolov5.draw(m, objects, options.classes);
            auto duration = duration_cast<milliseconds>(stop - start);
            float fps = 1000.0 / duration.count();
            std::cout << "Estimated frames per second : " << fps << ":" << duration.count() << std::endl;
            std::cout << "Objects: " << objects.size() << std::endl;
            for (auto obj : objects)
            {
                std::cout << "\t" << options.classes[obj.label] << ":" << obj.prob << " [" << obj.rect.x << ", " << obj.rect.y << ", " << obj.rect.width << ", " << obj.rect.height << "]" << std::endl;
            }
            if (options.save_result)
                writer.write(m);
            if (options.show_result)
                cv::imshow("sscma yolov5", m);
            if (cv::waitKey(1) == 27)
                break; // stop capturing by pressing ESC
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
            yolov5.detect(m, objects, options.score_threshold, options.iou_threshold);
            auto stop = high_resolution_clock::now();
            yolov5.draw(m, objects, options.classes);
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
                cv::imshow("sscma yolov5", m);
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