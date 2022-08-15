# mwe_hipsycl_hipfft 
How to run:
```
cd ./mwe_hipsycl_cufft/build
cmake .. -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;cuda:sm_75" && make && ./mwe-hipsycl-fft
<!-- https://github.com/ROCmSoftwarePlatform/hipFFT -->
<!-- cmake .. -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;cuda:sm_75" && make && ./mwe-hipsycl-fft -->
<!-- cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -DBUILD_WITH_LIB=CUDA -L -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;cuda:sm_75" && make && ./mwe-hipsycl-fft -->
<!-- cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -DBUILD_WITH_LIB=CUDA -L -->
<!-- HIP_PLATFORM=nvidia cmake .. -DBUILD_CLIENTS=ON -L -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;cuda:sm_75" && make && ./mwe-hipsycl-fft -->
<!-- HIP_PLATFORM=nvidia cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L -->
```
