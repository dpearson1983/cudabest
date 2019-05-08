#include <vector>
#include <vector_types.h>
#include <cufft.h>
#include <cmath>
#include "../include/cudabest.h"

__device__ void cudabest::swapIfGreater(double &a, double &b) {
    if (a > b) {
        double temp = a;
        a = b;
        b = temp;
    }
}

__device__ int cudabest::getBispecBin(double k1, double k2, double k3, double Delta_k, int numBins, 
                                      double k_min) {
    cudabest::swapIfGreater(k1, k2);
    cudabest::swapIfGreater(k1, k3);
    cudabest::swapIfGreater(k2, k3);
