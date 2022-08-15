# mwe_hipsycl_cufft 
Running these commands:
```
cd ./mwe_hipsycl_cufft/build
cmake .. -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;cuda:sm_75" && make && ./mwe-hipsycl-fft
```
