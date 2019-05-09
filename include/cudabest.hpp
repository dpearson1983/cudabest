#ifndef _CUDABEST_HPP_
#define _CUDABEST_HPP_

#include <vector>
#include <vector_types.h>
#include <cuda.h>
#include <cufft.h>

class cudabest{
    int4 N; // Grid dimensions
    double3 L, Delta_r, k_f, r_min; // Box size / Grid cell size / Fundamental frequencies
    int N_threads, N_blocks;
    double k_min, k_max;
    std::vector<int4> k_vecs; // Component frequencies and flattened array index for grid points in range
    std::vector<double3> k; // Frequency bins defined by k_min, k_max and Delta_k
            
    public:
        cufftDoubleComplex *d_F0, *d_F2, *d_Bij;
        unsigned long long int *d_Ntri;
        int4 *d_kvecs;
        
        cudabest(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double x_min, double y_min, 
                 double z_min, double k_min, double k_max);
        
        void getBispectrum(std::vector<double3> &gals, std::vector<double3> &rans, std::vector<double> &B_0,
                           std::vector<double> &B_2);
        
};

#endif
    
    
