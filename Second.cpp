#define NOMINMAX 
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "fft.h"
#include "utils.h"
#include "tests.h"

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define PATH_SEPARATOR "\\"
#else
#include <sys/stat.h>
#include <unistd.h>
#define PATH_SEPARATOR "/"
#endif




int main(int argc, char* argv[]) {

#ifdef _WIN32
    system("chcp 65001 >nul");
#endif

    //runTest1D();
    r//unTest1D_StepFunction();
   
    //runSinusoidImageTest();
    //runStripeImageTest();
    //runGaussian2DTest();
    //runDiagonalStripeTest();
    
    //Только прямое преобразование
    std::string inputPath = argv[1];
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);

    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "output_results" + PATH_SEPARATOR;

    if(!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        exit(1);
    }

    try {
        std::cout << "Size:" << image.cols << "x" << image.rows << " px\n";

        ImageUtils::saveImage(image, outputDir + "input_image.png");

        FFT::ComplexMatrix complexImage = ImageUtils::imageToComplexMatrix(image);
        FFT::ComplexMatrix fftResult = FFT::fft2D(complexImage, 1);
        FFT::ComplexMatrix shiftedFFT = FFT::shiftZeroFrequencyToCenter(fftResult);
        auto spectrum = FFT::computeLogAmplitudeSpectrum(shiftedFFT);
        cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrum);

        ImageUtils::saveImage(spectrumImage, outputDir + "output_image.png");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    /*
    //Прмямое + обратное преобразование
    std::string inputPath = argv[1];
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);

    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "output_results_restored" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        exit(1);
    }

    try {

        int originalHeight = static_cast<int>(image.rows);
        int originalWidth = static_cast<int>(image.cols);

        //Прямое преобразование
        ImageUtils::saveImage(image, outputDir + "input_image.png");

        FFT::ComplexMatrix complexImage = ImageUtils::imageToComplexMatrix(image);
        FFT::ComplexMatrix fftResult = FFT::fft2D(complexImage, 1);
        FFT::ComplexMatrix shiftedFFT = FFT::shiftZeroFrequencyToCenter(fftResult);


        auto spectrum = FFT::computeLogAmplitudeSpectrum(shiftedFFT);
        cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrum);
        ImageUtils::saveImage(spectrumImage, outputDir + "spectrum.png");


        // Обратное преобразование
        FFT::ComplexMatrix unshiftedFFT = FFT::shiftZeroFrequencyToCenter(shiftedFFT);

        FFT::ComplexMatrix restoredComplex = FFT::fft2D(unshiftedFFT, -1);

        FFT::ComplexMatrix croppedComplex(originalHeight, std::vector<FFT::Complex>(originalWidth));

        for (int y = 0; y < originalHeight; ++y) {
            for (int x = 0; x < originalWidth; ++x) {
                croppedComplex[y][x] = restoredComplex[y][x];
            }
        }

        cv::Mat restoredImage(originalHeight, originalWidth, CV_8UC1);
        for (int y = 0; y < originalHeight; ++y) {
            for (int x = 0; x < originalWidth; ++x) {
                double val = croppedComplex[y][x].real();
                val = std::max(0.0, std::min(255.0, val));
                restoredImage.at<uchar>(y, x) = static_cast<uchar>(val);
            }
        }

        ImageUtils::saveImage(restoredImage, outputDir + "output_image.png");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    */
    
    return 0;
}

