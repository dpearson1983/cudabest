include makefile.cudacompute

CXX = cuda-g++
VXX = nvcc $(ARCHS) -ccbin=$(CXX)
CXXFLAGS = -march=native -mtune=native -O3 -o -fPIC
VXXFLAGS = -lineinfo --compiler-options "$(CXXFLAGS)" -O3
LDFLAGS = -lcufft

library: obj/cudabest.o
	$(VXX) $(VXXFLAGS) $(LDFLAGS) $^ -fPIC -shared -o libcudabest.so
	
obj/cudabest.o: source/cudabest.cu
	$(VXX) $(VXXFLAGS) -dc source/cudabest.cu -o obj/cudabest.o
