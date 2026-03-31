#pragma once
#ifndef TESTS_H
#define TESTS_H


void runSinusoidImageTest();
void runStripeImageTest();
void runGaussian2DTest();
void runDiagonalStripeTest();
void runTest1D();

void saveSignalGraph(const std::vector<double>& signal, const std::string& filename);

void runTest1D_StepFunction();

#endif 