#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <regex>
#include <getopt.h>

class Options
{
public:
    enum InputType
    {
        Invalid,
        Image,
        Video,
        Camera
    };

    std::string model_path;
    std::string weights_path;
    std::string input_path;
    std::string output_path;
    int thread_num;
    bool save_result;
    bool show_result;
    bool classes_specified;
    std::vector<std::string> classes;
    std::vector<float> input_mean;
    std::vector<float> input_std;
    std::vector<int> input_shape;
    InputType input_type;
    float score_threshold;
    float iou_threshold;

    Options();
    Options(int thread_num, bool save_result, bool show_result, std::string model_path, std::string weights_path, std::vector<std::string> classes);
    Options(std::string model_path, std::string weights_path, std::vector<std::string> classes);
    void Parser(int argc, char *argv[]);
    void print_help();

private:
    void parse_float_arguments(const std::string &arg, std::vector<float> &output);
    void parse_int_arguments(const std::string &arg, std::vector<int> &output);
    void parse_string_arguments(const std::string &arg, std::vector<std::string> &output);
    enum InputType check_input_type(const std::string &input_path);
};

Options::Options()
{
    this->thread_num = 1;
    this->iou_threshold = 0.45;
    this->score_threshold = 0.25;
    this->save_result = false;
    this->show_result = true;
    this->classes_specified = false;
}

Options::Options(std::string model_path, std::string weights_path, std::vector<std::string> classes) : Options()
{
    this->model_path = model_path;
    this->weights_path = weights_path;
    this->classes = classes;
}

Options::Options(int thread_num, bool save_result, bool show_result, std::string model_path, std::string weights_path, std::vector<std::string> classes) : Options(model_path, weights_path, classes)
{
    this->thread_num = thread_num;
    this->save_result = save_result;
    this->show_result = show_result;
}

// parse ',' separated float arguments
void Options::parse_float_arguments(const std::string &arg, std::vector<float> &output)
{
    std::stringstream ss(arg);
    float value;
    output.clear();
    while (ss >> value)
    {
        output.push_back(value);
        if (ss.peek() == ',')
            ss.ignore();
    }
}

// parse ',' separated int arguments
void Options::parse_int_arguments(const std::string &arg, std::vector<int> &output)
{
    std::stringstream ss(arg);
    int value;
    output.clear();
    while (ss >> value)
    {
        output.push_back(value);
        if (ss.peek() == ',')
            ss.ignore();
    }
}

enum Options::InputType Options::check_input_type(const std::string &input_path)
{
    if (input_path.empty())
        return InputType::Invalid;
    std::regex image_regex(".*\\.(jpg|jpeg|png|gif|bmp|tiff|tif)$", std::regex::icase);
    std::regex video_regex(".*\\.(avi|mp4|mkv|wmv|flv|mov|webm)$", std::regex::icase);
    std::regex camera_regex("^[0-9]+$");
    if (std::regex_match(input_path, image_regex))
        return InputType::Image;
    if (std::regex_match(input_path, video_regex))
        return InputType::Video;
    if (std::regex_match(input_path, camera_regex))
        return InputType::Camera;
    return InputType::Invalid;
}

// parse ',' separated string arguments
void Options::parse_string_arguments(const std::string &arg, std::vector<std::string> &output)
{
    std::stringstream ss(arg);
    std::string value;
    output.clear();
    while (ss >> value)
    {
        output.push_back(value);
        if (ss.peek() == ',')
            ss.ignore();
    }
}

void Options::Parser(int argc, char *argv[])
{
    int opt;
    const char *const short_opts = "m:w:i:o:t:shc:";
    const option long_opts[] = {
        {"model", required_argument, nullptr, 'm'},
        {"weights", required_argument, nullptr, 'w'},
        {"input", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"thread", required_argument, nullptr, 't'},
        {"iou", required_argument, nullptr, 'u'},
        {"score", required_argument, nullptr, 'r'},
        {"save", no_argument, nullptr, 's'},
        {"headless", no_argument, nullptr, 'h'},
        {"classes", required_argument, nullptr, 'c'},
        {"input_mean", required_argument, nullptr, 'n'},
        {"input_std", required_argument, nullptr, 'd'},
        {"input_shape", required_argument, nullptr, 's'},
        {nullptr, no_argument, nullptr, 0}};

    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'm':
            model_path = optarg;
            break;
        case 'w':
            weights_path = optarg;
            break;
        case 'i':
            input_path = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 't':
            thread_num = std::stoi(optarg);
            break;
        case 'u':
            iou_threshold = std::stof(optarg);
            break;
        case 'r':
            score_threshold = std::stof(optarg);
            break;
        case 's':
            save_result = true;
            break;
        case 'h':
            show_result = false;
            break;
        case 'c':
            if (!classes_specified)
            { // overwrite default classes
                classes_specified = true;
                classes.clear();
            }
            parse_string_arguments(optarg, classes);
            break;
        case 'n':
            parse_float_arguments(optarg, input_mean);
            break;
        case 'd':
            parse_float_arguments(optarg, input_std);
            break;
        case 'p':
            parse_int_arguments(optarg, input_shape);
            break;
        default:
            std::cerr << "\x1b[31m"
                      << "Invalid argument"
                      << "\x1b[0m" << std::endl;
            print_help();
            exit(EXIT_FAILURE);
        }
    }

    if (model_path.empty() || weights_path.empty() || input_path.empty())
    { // check required arguments
        std::cerr << "\x1b[31m"
                  << "Missing required arguments"
                  << "\x1b[0m" << std::endl;
        print_help();
        exit(EXIT_FAILURE);
    }

    if ((input_type = check_input_type(input_path)) == InputType::Invalid)
    {
        std::cerr << "\x1b[31m"
                  << "Invalid input file type"
                  << "\x1b[0m" << std::endl;
        print_help();
        exit(EXIT_FAILURE);
    }

    if (output_path.empty() && save_result)
    { // set default output path
        if (input_type == InputType::Camera)
        {
            output_path = "camera_result.mp4";
        }
        else
        {
            size_t dot_pos = input_path.find_last_of('.');
            if (dot_pos == std::string::npos)
            {
                std::cerr << "\x1b[31m"
                          << "Invalid input file name"
                          << "\x1b[0m" << std::endl;
                print_help();
                exit(EXIT_FAILURE);
            }
            std::string prefix = input_path.substr(0, dot_pos);
            std::string suffix = input_path.substr(dot_pos);
            output_path = prefix + "_result" + suffix;
        }
    }

    if (input_mean.empty())
    {
        input_mean.push_back(0.0);
        input_mean.push_back(0.0);
        input_mean.push_back(0.0);
    }

    if (input_std.empty())
    {
        input_std.push_back(1.0 / 255);
        input_std.push_back(1.0 / 255);
        input_std.push_back(1.0 / 255);
    }

    if (input_shape.empty())
    {
        input_shape.push_back(1);
        input_shape.push_back(3);
        input_shape.push_back(640);
        input_shape.push_back(640);
    }

    std::cout << "model_path: " << model_path << std::endl;
    std::cout << "weights_path: " << weights_path << std::endl;
    std::cout << "input_path: " << input_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "input_type: " << input_type << std::endl;
    std::cout << "thread_num: " << thread_num << std::endl;
    std::cout << "iou_threshold: " << iou_threshold << std::endl;
    std::cout << "score_threshold: " << score_threshold << std::endl;
    std::cout << "save_result: " << save_result << std::endl;
    std::cout << "show_result: " << show_result << std::endl;
    std::cout << "input_mean: ";
    for (const auto &m : input_mean)
        std::cout << m << " ";
    std::cout << std::endl;
    std::cout << "input_std: ";
    for (const auto &s : input_std)
        std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "input_shape: ";
    for (const auto &s : input_shape)
        std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "classes: ";
    for (const auto &c : classes)
        std::cout << c << " ";
    std::cout << std::endl;
}

void Options::print_help()
{
    std::cout << "Usage: program_name --model=model_path --weights=weights_path --input=input_path --output=output_path [--thread=thread_num] [--save] [--show] [--classes=class1,class2,class3]..." << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model=model_path         Path to model file (default: " << model_path << ")" << std::endl;
    std::cout << "  --weights=weights_path     Path to weights file (default: " << weights_path << ")" << std::endl;
    std::cout << "  --input=input_path         Path to input file" << std::endl;
    std::cout << "  --output=output_path       Path to output file" << std::endl;
    std::cout << "  --thread=thread_num        Number of threads (default: 1)" << std::endl;
    std::cout << "  --iou=iou_threshold        IoU threshold (default: 0.45)" << std::endl;
    std::cout << "  --score=score_threshold    Score threshold (default: 0.25)" << std::endl;
    std::cout << "  --save                     Save result to file (default: false)" << std::endl;
    std::cout << "  --headless                 Do not show result (default: false)" << std::endl;
    std::cout << "  --classes=class1           List of classes to detect (default: all classes)" << std::endl;
    std::cout << "  --input_mean=0,0,0         Input mean (default: 0,0,0)" << std::endl;
    std::cout << "  --input_std=1,1,1          Input std (default: 0.0039126,0039126,0039126)" << std::endl;
    std::cout << "  --input_shape=1,3,640,640  Input shape (default: 1,3,640,640)" << std::endl;
}