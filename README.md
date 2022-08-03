# mwe_hipsycl_hipfft 
How to run:
```
cd ./mwe_hipsycl_cufft/build
cmake .. -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;hip:gfx906" && make && ./mwe-hipsycl-fft
```
