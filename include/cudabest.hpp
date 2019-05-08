#ifndef _CUDABEST_HPP_
#define _CUDABEST_HPP_

#include <vector>
#include <vector_types.h>

class cudabest{
    int4 N; // Grid dimensions
    double3 L, Delta_r, k_f; // Box size / Grid cell size / Fundamental frequencies
    int N_threads, N_blocks;
    double k_min, k_max, Delta_k;
    std::vector<int4> k_vecs; // Component frequencies and flattened array index for grid points in range
    int4 *d_kvecs; // Device storage for the above
    std::vector<double3> k; // Frequency bins defined by k_min, k_max and Delta_k
    double *d_F0, *d_F2, *d_Bij;
    unsigned long long int *d_Ntri;
    
    __device__ void swapIfGreater(double &a, double &b);
    
    __device__ int getBispecBin(double k1, double k2, double k3, double Delta_k, int numBins, double k_min);
    
    __global__ void zeroArrays(double *d_F0, double *d_F2, double *d_Bij, int4 N);
    
    __global__ void calculateNumTriangles(int4 *d_kvecs, unsigned long long int *d_Ntri, int N_kvecs, int4 N);
    
    __global__ void calculateBispectrum(double *d_F0 double *d_F2, double *d_Bij, int4 N, double *B0,
                                        double *B2);
    
    __global__ void bin(double3 *pos, double *d_F, double3 r_min, double3 Delta_r);
    
    public:
        
        cudabest(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double x_min, double y_min, 
                 double z_min, double k_min, double k_max);
        
        void getBispectrum(std::vector<double3> &gals, std::vector<double3> &rans, std::vector<double> &B_0,
                           std::vector<double> &B_2);
        
};

#endif
    
    
