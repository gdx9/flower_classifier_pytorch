#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

constexpr char kDnnName[] = "model_flowers.onnx";
constexpr char kImagePath[] = "flower_photos/daisy/5547758_eea9edfd54_n.jpg";
constexpr size_t kClassNumber = 5;
constexpr float kNormalizationRation = 1.f/255.f;

int main(){
    cv::dnn::Net model = cv::dnn::readNetFromONNX(kDnnName);

    // get mat
    cv::Mat image = cv::imread(kImagePath);

    // mat to blob
    cv::Mat blob = cv::dnn::blobFromImage(image,
        kNormalizationRation,
        {180,180});

    auto start = std::chrono::steady_clock::now();
    model.setInput(blob);

    cv::Mat output = model.forward();
    float* ptr = output.ptr<float>();

    auto end = std::chrono::steady_clock::now();
    double duration = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        ) / 1000.;// mcs to ms
    std::cout << "ONNX model's execution time: "<< duration << " milliseconds" << std::endl;

    for(size_t i = 0; i < kClassNumber; ++i)
        std::cout << ptr[i] << "  ";

    std::cout << std::endl << "end"<< std::endl;

    return 0;
}
