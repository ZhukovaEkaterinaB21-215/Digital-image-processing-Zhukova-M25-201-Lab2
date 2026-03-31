#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include "fft.h"

std::string getExecutablePath();

bool createDirectory(const std::string& path);

class ImageUtils {
public:
    static void saveImage(const cv::Mat& image, const std::string& filename);

    static FFT::ComplexMatrix imageToComplexMatrix(const cv::Mat& image);

    static cv::Mat spectrumToImage(const std::vector<std::vector<double>>& spectrum);

    static cv::Mat createSinusoidTestImage(int width, int height,
        const std::vector<std::pair<double, double>>& frequencies);

    static std::vector<FFT::Complex> createSinusoidTestSignal(int size,
        const std::vector<double>& frequencies);

    static void ensureDirectoryExists(const std::string& path);
};

#endif // UTILS_H