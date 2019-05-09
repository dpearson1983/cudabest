ARCHS = -gencode arch=compute_61,code=sm_61

CXX = cuda-g++
VXX = nvcc $(ARCHS) -ccbin=$(CXX)
CXXFLAGS = -march=native -mtune=native -O3 -fPIC
VXXFLAGS = -Xptxas -lineinfo --compiler-options "$(CXXFLAGS)" -O3
LDFLAGS = -lcufft -lstdc++

library: obj/cudabest.o
	$(VXX) $(VXXFLAGS) $(LDFLAGS) $^ -shared -o libcudabest.so
	
obj/cudabest.o: source/cudabest.cu
	$(VXX) $(VXXFLAGS) $(LDFLAGS) -dw source/cudabest.cu -o obj/cudabest.o
