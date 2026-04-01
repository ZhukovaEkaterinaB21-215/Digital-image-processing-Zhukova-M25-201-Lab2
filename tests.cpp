#define NOMINMAX 
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
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


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void runSinusoidImageTest() {
    const int WIDTH = 512, HEIGHT = 512;
    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "tests_results" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        exit(1);
    }

    std::vector<std::pair<double, double>> frequencies = {
        {10, 0}, {0, 16}, {8, 8}
    };

    cv::Mat testImage = ImageUtils::createSinusoidTestImage(WIDTH, HEIGHT, frequencies);
    ImageUtils::saveImage(testImage, outputDir + "input_sinusoids.png");


    FFT::ComplexMatrix complexImage = ImageUtils::imageToComplexMatrix(testImage);
    FFT::ComplexMatrix fftResult = FFT::fft2D(complexImage, 1);
    FFT::ComplexMatrix shiftedFFT = FFT::shiftZeroFrequencyToCenter(fftResult);
    auto spectrum = FFT::computeLogAmplitudeSpectrum(shiftedFFT);
    cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrum);
    ImageUtils::saveImage(spectrumImage, outputDir + "output_spectrum_sinusoids.png");

    int centerI = static_cast<int>(spectrum.size()) / 2;
    int centerJ = static_cast<int>(spectrum[0].size()) / 2;

    struct Peak { int x, y; int intensity; };

    int peakCount = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                ++peakCount;
            }
        }
    }

    std::vector<Peak> peaks(peakCount);

    int index = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                peaks[index].x = static_cast<int>(j) - centerJ;
                peaks[index].y = static_cast<int>(i) - centerI;
                peaks[index].intensity = static_cast<int>(spectrum[i][j]);
                ++index;
            }
        }
    }

    {
        std::ofstream f(outputDir + "peaks_sinusoids.txt");
        for (const auto& p : peaks) {
            f << p.x << " " << p.y << " " << p.intensity << "\n";
        }
    }

    {
        std::ofstream f(outputDir + "params_sinusoids.txt");
        f << WIDTH << " " << HEIGHT << "\n";
        for (const auto& freq : frequencies) {
            f << freq.first << " " << freq.second << "\n";
        }
    }
}


void runStripeImageTest() {
    const int WIDTH = 512, HEIGHT = 512;
    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "tests_results" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        exit(1);
    }

    int stripeThickness = 40; 
    int stripeStart = 100;
    int stripeEnd = 400; 
    int stripeCenterX = WIDTH / 2;

    cv::Mat testImage = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);

   
    int leftX = stripeCenterX - stripeThickness / 2;
    int rightX = stripeCenterX + stripeThickness / 2;

    for (int y = stripeStart; y < stripeEnd; ++y) {
        for (int x = leftX; x < rightX; ++x) {
            testImage.at<uchar>(y, x) = 255;
        }
    }

    ImageUtils::saveImage(testImage, outputDir + "input_stripe.png");


    FFT::ComplexMatrix complexImage = ImageUtils::imageToComplexMatrix(testImage);
    FFT::ComplexMatrix fftResult = FFT::fft2D(complexImage, 1);
    FFT::ComplexMatrix shiftedFFT = FFT::shiftZeroFrequencyToCenter(fftResult);
    auto spectrum = FFT::computeLogAmplitudeSpectrum(shiftedFFT);
    cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrum);
    ImageUtils::saveImage(spectrumImage, outputDir + "output_spectrum_stripe.png");

    int centerI = static_cast<int>(spectrum.size()) / 2;
    int centerJ = static_cast<int>(spectrum[0].size()) / 2;

    struct Peak { int x, y; int intensity; };

    int peakCount = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                ++peakCount;
            }
        }
    }

    std::vector<Peak> peaks(peakCount);

    int index = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                peaks[index].x = static_cast<int>(j) - centerJ;
                peaks[index].y = static_cast<int>(i) - centerI;
                peaks[index].intensity = static_cast<int>(spectrum[i][j]);
                ++index;
            }
        }
    }

    {
        std::ofstream f(outputDir + "peaks_white_stripe.txt");
        for (const auto& p : peaks) {
            f << p.x << " " << p.y << " " << p.intensity << "\n";
        }
    }

    {
        std::ofstream f(outputDir + "params_white_stripe.txt");
        f << WIDTH << " " << HEIGHT << "\n";
        f << stripeThickness << " " << stripeStart << " " << stripeEnd << "\n";
    }
}

void runGaussian2DTest() {
    const int WIDTH = 512, HEIGHT = 512;
    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "tests_results" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        exit(1);
    }

    double sigma = 1.0;
    int centerX = WIDTH / 2;
    int centerY = HEIGHT / 2;
    double amplitude = 255.0;


    cv::Mat testImage = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);

    double twoSigmaSq = 2.0 * sigma * sigma;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double dx = static_cast<double>(x - centerX);
            double dy = static_cast<double>(y - centerY);
            double distSq = dx * dx + dy * dy;

            double exponent = -distSq / twoSigmaSq;
            double value = amplitude * std::exp(exponent);

            testImage.at<uchar>(y, x) = static_cast<uchar>(std::max(0.0, std::min(255.0, value)));
        }
    }

    ImageUtils::saveImage(testImage, outputDir + "input_gaussian2d.png");


    FFT::ComplexMatrix complexImage = ImageUtils::imageToComplexMatrix(testImage);
    FFT::ComplexMatrix fftResult = FFT::fft2D(complexImage, 1);
    FFT::ComplexMatrix shiftedFFT = FFT::shiftZeroFrequencyToCenter(fftResult);
    auto spectrum = FFT::computeLogAmplitudeSpectrum(shiftedFFT);
    cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrum);
    ImageUtils::saveImage(spectrumImage, outputDir + "output_spectrum_gaussian2d.png");

    int centerI = static_cast<int>(spectrum.size()) / 2;
    int centerJ = static_cast<int>(spectrum[0].size()) / 2;

    struct Peak { int x, y; int intensity; };

    int peakCount = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                ++peakCount;
            }
        }
    }

    std::vector<Peak> peaks(peakCount);

    int index = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                peaks[index].x = static_cast<int>(j) - centerJ;
                peaks[index].y = static_cast<int>(i) - centerI;
                peaks[index].intensity = static_cast<int>(spectrum[i][j]);
                ++index;
            }
        }
    }

    {
        std::ofstream f(outputDir + "peaks_gaussian2d.txt");
        for (const auto& p : peaks) {
            f << p.x << " " << p.y << " " << p.intensity << "\n";
        }
    }

    {
        std::ofstream f(outputDir + "params_gaussian2d.txt");
        f << WIDTH << " " << HEIGHT << "\n";
        f << sigma << " " << centerX << " " << centerY << " " << amplitude << "\n";
    }
}


void runDiagonalStripeTest() {
    const int WIDTH = 512, HEIGHT = 512;
    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "tests_results" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        exit(1);
    }

    int stripeThickness = 30;
    double angleDegrees = 45.0;
    double angleRadians = angleDegrees * M_PI / 180.0;
    int centerX = WIDTH / 2; 
    int centerY = HEIGHT / 2; 

    int stripeLength = 250;
    int halfLength = stripeLength / 2;     

    cv::Mat testImage = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double dx = static_cast<double>(x - centerX);
            double dy = static_cast<double>(y - centerY);
            double distPerp = std::abs(dx * std::sin(angleRadians) - dy * std::cos(angleRadians));
            double distParallel = dx * std::cos(angleRadians) + dy * std::sin(angleRadians);
            if (distPerp < stripeThickness / 2.0 &&
                std::abs(distParallel) < halfLength) {
                testImage.at<uchar>(y, x) = 255;
            }
        }
    }

    ImageUtils::saveImage(testImage, outputDir + "input_diagonal_stripe.png");


    FFT::ComplexMatrix complexImage = ImageUtils::imageToComplexMatrix(testImage);
    FFT::ComplexMatrix fftResult = FFT::fft2D(complexImage, 1);
    FFT::ComplexMatrix shiftedFFT = FFT::shiftZeroFrequencyToCenter(fftResult);
    auto spectrum = FFT::computeLogAmplitudeSpectrum(shiftedFFT);
    cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrum);
    ImageUtils::saveImage(spectrumImage, outputDir + "output_spectrum_diagonal_stripe.png");

    int centerI = static_cast<int>(spectrum.size()) / 2;
    int centerJ = static_cast<int>(spectrum[0].size()) / 2;

    struct Peak { int x, y; int intensity; };

    int peakCount = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                ++peakCount;
            }
        }
    }

    std::vector<Peak> peaks(peakCount);

    int index = 0;
    for (size_t i = 0; i < spectrum.size(); ++i) {
        for (size_t j = 0; j < spectrum[0].size(); ++j) {
            if (spectrum[i][j] > 200) {
                peaks[index].x = static_cast<int>(j) - centerJ;
                peaks[index].y = static_cast<int>(i) - centerI;
                peaks[index].intensity = static_cast<int>(spectrum[i][j]);
                ++index;
            }
        }
    }

    {
        std::ofstream f(outputDir + "peaks_diagonal_stripe.txt");
        for (const auto& p : peaks) {
            f << p.x << " " << p.y << " " << p.intensity << "\n";
        }
    }

    {
        std::ofstream f(outputDir + "params_diagonal_stripe.txt");
        f << WIDTH << " " << HEIGHT << "\n";
        f << stripeThickness << " " << stripeLength << " " << angleDegrees << " " << centerX << " " << centerY << "\n";
    }
}


void drawLine(cv::Mat& image, int x0, int y0, int x1, int y1, const cv::Vec3b& color) {
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if (x0 >= 0 && x0 < image.cols && y0 >= 0 && y0 < image.rows) {
            image.at<cv::Vec3b>(y0, x0) = color;
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

void saveSignalGraph(const std::vector<double>& signal, const std::string& filename) {
    
    const int WIDTH = static_cast<int>(signal.size());
    const int HEIGHT = 600;
    const int PADDING = 20;

    cv::Mat image(HEIGHT, WIDTH, CV_8UC3);


    double minVal = signal[0], maxVal = signal[0];
    for (double v : signal) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    double range = maxVal - minVal;
    if (range < 1e-10) range = 1.0;

    const cv::Vec3b COLOR_BG(255, 255, 255);
    const cv::Vec3b COLOR_AXIS(200, 200, 200);
    const cv::Vec3b COLOR_SIGNAL(255, 0, 0); 

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            image.at<cv::Vec3b>(y, x) = COLOR_BG;
        }
    }

    int axisY = HEIGHT - PADDING;
    for (int x = 0; x < WIDTH; ++x) {
        if (axisY >= 0 && axisY < HEIGHT) {
            image.at<cv::Vec3b>(axisY, x) = COLOR_AXIS;
        }
    }

    std::vector<std::pair<int, int>> points;
    points.reserve(WIDTH);

    for (int x = 0; x < WIDTH; ++x) {
        double normalized = (signal[x] - minVal) / range;
        int y = static_cast<int>((1.0 - normalized) * (HEIGHT - 2 * PADDING) + PADDING);
        y = std::max(0, std::min(HEIGHT - 1, y));
        points.emplace_back(x, y);
    }

    for (int i = 0; i < WIDTH - 1; ++i) {
        drawLine(image,
            points[i].first, points[i].second,
            points[i + 1].first, points[i + 1].second,
            COLOR_SIGNAL);
    }

    ImageUtils::saveImage(image, filename);
}



void runTest1D() {
    const int SAMPLE_RATE = 512;
    const double DURATION = 1.0;
    const int N = static_cast<int>(SAMPLE_RATE * DURATION);

    std::vector<std::pair<double, double>> sinusoids = {
        {10.0, 20.0}
    };

    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "tests_results" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        return;
    }

    FFT fft;
    std::vector<FFT::Complex> signal(N);
    double t_step = 1.0 / SAMPLE_RATE;

    for (int i = 0; i < N; ++i) {
        double t = i * t_step;
        double val = 0.0;
        for (const auto& [freq, amp] : sinusoids) {
            val += amp * std::sin(2.0 * M_PI * freq * t);
        }
        signal[i] = FFT::Complex(val, 0.0);
    }
 
    std::vector<double> signalReal(N);
    for (int i = 0; i < N; ++i) {
        signalReal[i] = signal[i].real();
    }
    saveSignalGraph(signalReal, outputDir + "test_01_signal_graph.png");

    auto spectrum = fft.fft1D(signal, 1);
    int specSize = static_cast<int>(spectrum.size());
 
    std::vector<std::vector<double>> logSpectrum(1, std::vector<double>(specSize));
    double maxVal = 1e-10;

    for (int i = 0; i < specSize; ++i) {
        double amp = std::abs(spectrum[i]);
        logSpectrum[0][i] = std::log(1.0 + amp);
        if (logSpectrum[0][i] > maxVal) {
            maxVal = logSpectrum[0][i];
        }
    }

    for (int i = 0; i < specSize; ++i) {
        logSpectrum[0][i] = (logSpectrum[0][i] / maxVal) * 255.0;
    }

    const int VIS_HEIGHT = 100;
    std::vector<std::vector<double>> spectrumVis(VIS_HEIGHT, std::vector<double>(specSize));

    for (int r = 0; r < VIS_HEIGHT; ++r) {
        for (int c = 0; c < specSize; ++c) {
            spectrumVis[r][c] = logSpectrum[0][c];
        }
    }
    cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrumVis);
    ImageUtils::saveImage(spectrumImage, outputDir + "test_01_spectrum_1D.png");

    struct Peak { int index; double frequency; int intensity; };

    const double THRESHOLD = 200.0;
    const int NYQUIST = specSize / 2;

    int peakCount = 0;
    for (int i = 0; i < NYQUIST; ++i) {
        if (logSpectrum[0][i] > THRESHOLD) {
            ++peakCount;
        }
    }

    std::vector<Peak> peaks(peakCount);

    int index = 0;
    for (int i = 0; i < NYQUIST; ++i) {
        if (logSpectrum[0][i] > THRESHOLD) {
            double frequency = static_cast<double>(i) * SAMPLE_RATE / N;
            peaks[index].index = i;
            peaks[index].frequency = frequency;
            peaks[index].intensity = static_cast<int>(logSpectrum[0][i]);
            ++index;
        }
    }
    {
        std::ofstream f(outputDir + "test_01_peaks.txt");
        for (const auto& p : peaks) {
            f << p.index << " " << p.frequency << " " << p.intensity << "\n";
        }
        f.close();
    }
    {
        std::ofstream f(outputDir + "test_01_params.txt");
        f << "sample_rate " << SAMPLE_RATE << "\n";
        f << "duration " << DURATION << "\n";
        f << "signal_length " << N << "\n";
        f << "spectrum_length " << specSize << "\n";
        f << "sinusoids: frequency_Hz amplitude\n";
        for (const auto& [freq, amp] : sinusoids) {
            f << freq << " " << amp << "\n";
        }
        f.close();
    }
}

void runTest1D_StepFunction() {
    const int SAMPLE_RATE = 512;
    const double DURATION = 1.0;
    const int N = static_cast<int>(SAMPLE_RATE * DURATION);

    const double STEP_POSITION = 0.5;
    const double AMPLITUDE_HIGH = 1.0;
    const double AMPLITUDE_LOW = 0.0;

    int stepIndex = static_cast<int>(N * STEP_POSITION);

    std::string exePath = getExecutablePath();
    std::string outputDir = exePath + PATH_SEPARATOR + "tests_results" + PATH_SEPARATOR;

    if (!createDirectory(outputDir)) {
        std::cerr << "ERROR: Cannot create output directory: " << outputDir << "\n";
        return;
    }

    FFT fft;
    std::vector<FFT::Complex> signal(N);

    for (int i = 0; i < N; ++i) {
        double val = (i < stepIndex) ? AMPLITUDE_HIGH : AMPLITUDE_LOW;
        signal[i] = FFT::Complex(val, 0.0);
    }

    std::vector<double> signalReal(N);
    for (int i = 0; i < N; ++i) {
        signalReal[i] = signal[i].real();
    }
    saveSignalGraph(signalReal, outputDir + "test_01_falling_step_signal.png");

    auto spectrum = fft.fft1D(signal, 1);
    int specSize = static_cast<int>(spectrum.size());

    std::vector<std::vector<double>> logSpectrum(1, std::vector<double>(specSize));
    double maxVal = 1e-10;

    for (int i = 0; i < specSize; ++i) {
        double amp = std::abs(spectrum[i]);
        logSpectrum[0][i] = std::log(1.0 + amp);
        if (logSpectrum[0][i] > maxVal) {
            maxVal = logSpectrum[0][i];
        }
    }

    for (int i = 0; i < specSize; ++i) {
        logSpectrum[0][i] = (logSpectrum[0][i] / maxVal) * 255.0;
    }

    const int VIS_HEIGHT = 150;
    std::vector<std::vector<double>> spectrumVis(VIS_HEIGHT, std::vector<double>(specSize));

    for (int r = 0; r < VIS_HEIGHT; ++r) {
        for (int c = 0; c < specSize; ++c) {
            spectrumVis[r][c] = logSpectrum[0][c];
        }
    }

    cv::Mat spectrumImage = ImageUtils::spectrumToImage(spectrumVis);
    ImageUtils::saveImage(spectrumImage, outputDir + "test_01_falling_step_spectrum.png");

    struct Peak { int index; double frequency; int intensity; };

    const double THRESHOLD = 100.0;
    const int NYQUIST = specSize / 2;

    int peakCount = 0;
    for (int i = 0; i < NYQUIST; ++i) {
        if (logSpectrum[0][i] > THRESHOLD) {
            ++peakCount;
        }
    }

    std::vector<Peak> peaks(peakCount);

    int index = 0;
    for (int i = 0; i < NYQUIST; ++i) {
        if (logSpectrum[0][i] > THRESHOLD) {
            double frequency = static_cast<double>(i) * SAMPLE_RATE / N;
            peaks[index].index = i;
            peaks[index].frequency = frequency;
            peaks[index].intensity = static_cast<int>(logSpectrum[0][i]);
            ++index;
        }
    }
    {
        std::ofstream f(outputDir + "test_01_falling_step_peaks.txt");
        for (const auto& p : peaks) {
            f << p.index << " " << p.frequency << " " << p.intensity << "\n";
        }
        f.close();
    }
    {
        std::ofstream f(outputDir + "test_01_falling_step_params.txt");
        f << "signal_type falling_step\n";
        f << "sample_rate " << SAMPLE_RATE << "\n";
        f << "duration " << DURATION << "\n";
        f << "signal_length " << N << "\n";
        f << "step_position " << STEP_POSITION << "\n";
        f << "step_index " << stepIndex << "\n";
        f << "amplitude_high " << AMPLITUDE_HIGH << " (before step)\n";
        f << "amplitude_low " << AMPLITUDE_LOW << " (after step)\n";
        f.close();
    }
}