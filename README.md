# mwe_hipsycl_cufft 
Running these commands:
```
git clone 
cd ./mwe_hipsycl_cufft
cd build
cmake .. -DhipSYCL_DIR=/opt/hipSYCL/lib/cmake/hipSYCL/ -DHIPSYCL_TARGETS="omp;cuda:sm_75" && make && ./mwe-hipsycl-fft
```
Produces the following output:
```
-- The CXX compiler identification is Clang 14.0.6
-- The CUDA compiler identification is NVIDIA 11.7.64
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/clang++-14 - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda/include (found version "11.7.64") 
-- Looking for C++ include pthread.h
-- Looking for C++ include pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/boeck/mwe_hipSYCL_cufft/build
[ 50%] Building CXX object CMakeFiles/mwe-hipsycl-fft.dir/src/mwe.cpp.o
syclcc warning: No optimization flag was given, optimizations are disabled by default. Performance may be degraded. Compile with e.g. -O2/-O3 to enable optimizations.
clang: warning: CUDA version is newer than the latest supported version 11.5 [-Wunknown-cuda-version]
ptxas fatal   : Unresolved extern function 'cufftPlan3d'
clang: error: ptxas command failed with exit code 255 (use -v to see invocation)
Ubuntu clang version 14.0.6-++20220622053019+f28c006a5895-1~exp1~20220622173056.159
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/lib/llvm-14/bin
clang: note: diagnostic msg: 
********************

PLEASE ATTACH THE FOLLOWING FILES TO THE BUG REPORT:
Preprocessed source(s) and associated run script(s) are located at:
clang: note: diagnostic msg: /tmp/mwe-31a185.cu
clang: note: diagnostic msg: /tmp/mwe-9a429d/mwe-sm_75.cu
clang: note: diagnostic msg: /tmp/mwe-31a185.sh
clang: note: diagnostic msg: 

********************
CMakeFiles/mwe-hipsycl-fft.dir/build.make:77: recipe for target 'CMakeFiles/mwe-hipsycl-fft.dir/src/mwe.cpp.o' failed
make[2]: *** [CMakeFiles/mwe-hipsycl-fft.dir/src/mwe.cpp.o] Error 255
CMakeFiles/Makefile2:82: recipe for target 'CMakeFiles/mwe-hipsycl-fft.dir/all' failed
make[1]: *** [CMakeFiles/mwe-hipsycl-fft.dir/all] Error 2
Makefile:90: recipe for target 'all' failed
make: *** [all] Error 2
```
