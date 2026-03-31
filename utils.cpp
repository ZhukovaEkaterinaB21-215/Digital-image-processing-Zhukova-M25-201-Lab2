#define NOMINMAX 
#define _CRT_SECURE_NO_WARNINGS

#include "utils.h"
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define PATH_SEPARATOR "\\"
#else
#include <sys/stat.h>
#include <unistd.h>
#define PATH_SEPARATOR "/"
#endif

std::string getExecutablePath() {
    std::string path;
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    path = std::string(buffer);
    size_t pos = path.find_last_of("\\/");
    if (pos != std::string::npos) {
        path = path.substr(0, pos);
    }
#else
    char buffer[1024];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        path = std::string(buffer);
        size_t pos = path.find_last_of("/");
        if (pos != std::string::npos) {
            path = path.substr(0, pos);
        }
    }
#endif
    return path;
}

bool createDirectory(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}



void ImageUtils::saveImage(const cv::Mat& image, const std::string& filename) {
    bool saved = cv::imwrite(filename, image);
    if (saved) {
        std::cout << "  [SAVE] Image: " << filename << "\n";
    }
    else {
        std::cerr << "  [ERROR] Failed to save image: " << filename << "\n";
    }
}

FFT::ComplexMatrix ImageUtils::imageToComplexMatrix(const cv::Mat& image) {
    FFT::ComplexMatrix matrix(image.rows, std::vector<FFT::Complex>(image.cols));

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            matrix[i][j] = FFT::Complex(
                static_cast<double>(image.at<uchar>(i, j)), 0);
        }
    }

    return matrix;
}

cv::Mat ImageUtils::spectrumToImage(const std::vector<std::vector<double>>& spectrum) {
    if (spectrum.empty() || spectrum[0].empty()) {
        throw std::invalid_argument("Empty spectrum");
    }

    cv::Mat image(
        static_cast<int>(spectrum.size()),
        static_cast<int>(spectrum[0].size()),
        CV_8UC1);

    for (size_t i = 0; i < spectrum.size(); i++) {
        for (size_t j = 0; j < spectrum[0].size(); j++) {
            int val = static_cast<int>(std::round(spectrum[i][j]));
            image.at<uchar>(static_cast<int>(i), static_cast<int>(j)) =
                static_cast<uchar>(std::clamp(val, 0, 255));
        }
    }

    return image;
}

cv::Mat ImageUtils::createSinusoidTestImage(int width, int height,
    const std::vector<std::pair<double, double>>& frequencies) {

    cv::Mat image(height, width, CV_8UC1, cv::Scalar(128));
    const double PI = std::acos(-1.0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double value = 128.0;
            for (const auto& freq : frequencies) {
                double freqX = freq.first;
                double freqY = freq.second;
                value += 60.0 * std::sin(2.0 * PI * (freqX * x / width + freqY * y / height));
            }
            image.at<uchar>(y, x) = static_cast<uchar>(std::clamp(value, 0.0, 255.0));
        }
    }

    return image;
}

std::vector<FFT::Complex> ImageUtils::createSinusoidTestSignal(int size,
    const std::vector<double>& frequencies) {

    std::vector<FFT::Complex> signal(size);
    const double PI = std::acos(-1.0);

    for (int i = 0; i < size; i++) {
        double value = 0;
        for (double freq : frequencies) {
            value += std::sin(2.0 * PI * freq * i / size);
        }
        signal[i] = FFT::Complex(value, 0);
    }

    return signal;
}