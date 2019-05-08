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
    int i = (k1 - k_min)/Delta_k;
    int j = (k2 - k_min)/Delta_k;
    int k = (k3 - k_min)/Delta_k;
    int bin = k + numBins*(j + numBins*i);
    return bin;
}

__global__ void cudabest::zeroArrays(double *d_F0, double *d_F2, double *d_Bij, int4 N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (tid < N.w) {
        d_F0[tid] = 0.0;
        d_F2[tid] = 0.0;
        d_Bij[tid] = 0.0;
    }
}

__global__ void cudabest::calculateNumTriangles(
