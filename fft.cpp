#include "fft.h"
#include <stdexcept>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool FFT::isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int FFT::log2Int(int n) {
    int result = 0;
    while ((1 << result) < n) {
        ++result;
    }
    return result;
}

int FFT::nextPowerOfTwo(int n) {
    if (n <= 0) return 1;
    if (isPowerOfTwo(n)) return n;
    return 1 << log2Int(n);
}

void FFT::complexBitReverse(Complex* data, int size) {
    int middle = size / 2;
    int revSize = size - 1;
    int j = 0;

    for (int i = 0; i < revSize; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        int k = middle;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

void FFT::fft1D(Complex* data, int size, int dir) {
    if (!isPowerOfTwo(size)) {
        throw std::invalid_argument("The size must be a power of two");
    }

    if (size == 1) return;

    int sizeLog2 = log2Int(size);
    complexBitReverse(data, size);

    int ptsInLeftDft, ptsInRightDft = 1;
    for (int stage = 1; stage <= sizeLog2; ++stage) {
        ptsInLeftDft = ptsInRightDft;
        ptsInRightDft *= 2;

        Complex twiddle(1.0, 0.0);
        double trigArg = M_PI / ptsInLeftDft;
        Complex wFactor(std::cos(trigArg), -std::sin(trigArg) * dir);

        for (int butterflyPos = 0; butterflyPos < ptsInLeftDft; ++butterflyPos) {
            for (int topNode = butterflyPos; topNode < size; topNode += ptsInRightDft) {
                int botNode = topNode + ptsInLeftDft;
                Complex temp = data[botNode] * twiddle;
                data[botNode] = data[topNode] - temp;
                data[topNode] += temp;
            }
            twiddle *= wFactor;
        }
    }

    if (dir == -1) {
        for (int i = 0; i < size; ++i) {
            data[i] /= static_cast<double>(size);
        }
    }
}

std::vector<FFT::Complex> FFT::fft1D(const std::vector<Complex>& signal, int dir) {
    if (signal.empty()) {
        throw std::invalid_argument("Empty signal");
    }

    int originalSize = static_cast<int>(signal.size());
    int newSize = nextPowerOfTwo(originalSize);

    std::vector<Complex> padded(newSize, Complex(0, 0));
    for (int i = 0; i < originalSize; ++i) {
        padded[i] = signal[i];
    }

    fft1D(padded.data(), newSize, dir);
    return padded;
}

FFT::ComplexMatrix FFT::fft2D(const ComplexMatrix& image, int dir) {
    if (image.empty() || image[0].empty()) {
        throw std::invalid_argument("Blank image");
    }

    int rows = static_cast<int>(image.size());
    int cols = static_cast<int>(image[0].size());

    int newRows = nextPowerOfTwo(rows);
    int newCols = nextPowerOfTwo(cols);

    ComplexMatrix padded(newRows, std::vector<Complex>(newCols, Complex(0, 0)));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            padded[i][j] = image[i][j];
        }
    }

    for (int i = 0; i < newRows; ++i) {
        fft1D(padded[i].data(), newCols, dir);
    }

    std::vector<Complex> column(newRows);
    for (int j = 0; j < newCols; ++j) {
        for (int i = 0; i < newRows; ++i) {
            column[i] = padded[i][j];
        }
        fft1D(column.data(), newRows, dir);
        for (int i = 0; i < newRows; ++i) {
            padded[i][j] = column[i];
        }
    }

    return padded;
}

FFT::ComplexMatrix FFT::shiftZeroFrequencyToCenter(const ComplexMatrix& fftResult) {
    if (fftResult.empty() || fftResult[0].empty()) {
        throw std::invalid_argument("Empty FFT result");
    }

    int rows = static_cast<int>(fftResult.size());
    int cols = static_cast<int>(fftResult[0].size());

    ComplexMatrix shifted(rows, std::vector<Complex>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int newI = (i + rows / 2) % rows;
            int newJ = (j + cols / 2) % cols;
            shifted[newI][newJ] = fftResult[i][j];
        }
    }

    return shifted;
}

std::vector<std::vector<double>> FFT::computeLogAmplitudeSpectrum(
    const ComplexMatrix& fftResult) {

    if (fftResult.empty() || fftResult[0].empty()) {
        throw std::invalid_argument("Empty FFT result");
    }

    int rows = static_cast<int>(fftResult.size());
    int cols = static_cast<int>(fftResult[0].size());

    std::vector<std::vector<double>> spectrum(rows, std::vector<double>(cols));

    double maxVal = 1e-10;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double amplitude = std::abs(fftResult[i][j]);
            spectrum[i][j] = std::log(1.0 + amplitude);
            if (spectrum[i][j] > maxVal) {
                maxVal = spectrum[i][j];
            }
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            spectrum[i][j] = (spectrum[i][j] / maxVal) * 255.0;
        }
    }

    return spectrum;
}