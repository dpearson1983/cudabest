#include <iostream>
#include <vector>
#include <vector_types.h>
#include <cuda.h>
#include <cufft.h>
#include <cmath>
#include <cstddef>
#include "../include/cudabest.hpp"

__constant__ double2 d_klim;
__constant__ double d_Deltak;
__constant__ int4 d_N;
__constant__ double3 d_kf

__device__ void swapIfGreater(double &a, double &b) {
    if (a > b) {
        double temp = a;
        a = b;
        b = temp;
    }
}

__device__ int getBispecBin(double k1, double k2, double k3, int numBins) {
    swapIfGreater(k1, k2);
    swapIfGreater(k1, k3);
    swapIfGreater(k2, k3);
    int i = (k1 - d_klim.x)/d_Deltak;
    int j = (k2 - d_klim.x)/d_Deltak;
    int k = (k3 - d_klim.x)/d_Deltak;
    int bin = k + numBins*(j + numBins*i);
    return bin;
}

__global__ void zeroArrays(cufftDoubleComplex *dF0, cufftDoubleComplex *dF2, cufftDoubleComplex *dBij, 
                                     int4 N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (tid < N.w) {
        this->d_F0[tid] = 0.0;
        this->d_F2[tid] = 0.0;
        this->d_Bij[tid] = 0.0;
    }
}

__global__ void calculateNumTriangles(int4 *d_kvecs, double *k_mags, unsigned long long int *dNtri, 
                                                int N_kvecs, int N_bins) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int N_init = N_bins.w/blockDim.x + 1;
    int startInit = threadIdx.x*N_init;
    
    extern __shared__ unsigned long long int Ntri_local[];
    for (int i = startInit; i < startInit + N_init; ++i) {
        if (i < N_bins) {
            Ntri_local[i] = 0;
        }
    }
    __syncthreads();
    
    if (tid < N_kvecs) {
        int4 k_1 = d_kvecs[tid];
        double k_1mag = k_mags[tid];
        for (int i = tid; i < N_kvecs; ++i) {
            int4 k_2 = d_kvecs[i];
            double k_2mag = k_mags[i];
            int4 k_3 = {-k_1.x - k_2.x, -k_1.y - k_2.y, -k_1.z - k_2.z, 0};
            double3 k3 = {k_3.x*d_kf.x, k_3.y*d_kf.y, k_3.z*d_kf.z};
            double k_3mag = __dsqrt_rn(k3.x*k3.x + k3.y*k3.y + k3.z*k3.z);
            if (k_3mag >= d_klim.x && k_3mag < d_klim.y) {
                getBispecBin(k_1mag, k2_mag, k3_mag, N_bins);
                atomicAdd(&Ntri_local[bin], 1L);
            }
        }
    }
    __syncthreads();
    
    for (int i = startInit; i < startInit + N_init; ++i) {
        atomicAdd(dNtri[i], Ntri_local[i]);
    }
}

// __global__ void calculateBispectrum(int 4, double *B0, double *B2) {
//     int tid = threadIdx.x + blockDim.x*blockIdx.x;
// }
// 
// __global__ void bin(double3 *pos, double3 r_min, double3 Delta_r) {
//     int tid = threadIdx.x + blockDim.x*blockIdx.x;
// }

cudabest::cudabest(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double x_min, double y_min, 
                   double z_min, double k_min, double k_max, int N_bins) {
    double2 k_lim = {k_min, k_max};
    this->k_f = {2.0*M_PI/Lx, 2.0*M_PI/Ly, 2.0*M_PI/L.z};
    this->N = {Nx, Ny, Nz, 0};
    this->N.w = Nx*Ny*Nz;
    this->L = {Lx, Ly, Lz};
    this->Delta_r = {Lx/Nx, Ly/Ny, Lz/Nz};
    this->r_min = {x_min, y_min, z_min};
    this->Delta_k = (k_max - k_min)/N_bins;
    
    cudaMemcpyToSymbol(d_klim, &k_lim, sizeof(double2));
    cudaMemcpyToSymbol(d_kf, this->k_f, sizeof(double3));
    cudaMemcpyToSymbol(d_N, this->N, sizeof(int4));
    cudaMemcpyToSymbol(d_Deltak, this->Delta_k, sizeof(double));
    
    cudaMalloc((void **)&this->d_F0, this->N.w*sizeof(cufftDoubleComplex));
    cudaMalloc((void **)&this->d_F2, this->N.w*sizeof(cufftDoubleComplex));
    cudaMalloc((void **)&this->d_Bij, this->N.w*sizeof(cufftDoubleComplex));
    
    cudabest::zeroArrays(this->d_F0, this->d_F2, this->d_Bij, this->N);
}

void cudabest::getBispectrum(std::vector<double3> &gals, std::vector<double3> &rans, std::vector<double> &B_0,
                             std::vector<double> &B_2) {
    std::cout << "Not implemented." << std::endl;
}
