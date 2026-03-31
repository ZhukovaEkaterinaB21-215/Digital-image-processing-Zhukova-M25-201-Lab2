#ifndef FFT_H
#define FFT_H

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

class FFT {
public:
    using Complex = std::complex<double>;

    using ComplexMatrix = std::vector<std::vector<Complex>>;

    static void fft1D(Complex* data, int size, int dir = 1);

    static std::vector<Complex> fft1D(const std::vector<Complex>& signal, int dir = 1);

    static ComplexMatrix fft2D(const ComplexMatrix& image, int dir = 1);

    static std::vector<std::vector<double>>
        computeLogAmplitudeSpectrum(const ComplexMatrix& fftResult);

    static ComplexMatrix shiftZeroFrequencyToCenter(const ComplexMatrix& fftResult);

    static void complexBitReverse(Complex* data, int size);

    static int log2Int(int n);

    static bool isPowerOfTwo(int n);

    static int nextPowerOfTwo(int n);
};

#endif // FFT_H