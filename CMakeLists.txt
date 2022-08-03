cmake_minimum_required(VERSION 3.22)

# set(CMAKE_C_COMPILER "/usr/bin/clang-14")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-14")
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 75)
# endif()

project(mwe-hipsycl-cufft VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(SRC_PATH "src" CACHE PATH "Sources.")

set(MY_SOURCE_FILES
    ${SRC_PATH}/mwe.cpp)

find_package(hipSYCL REQUIRED)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

message(STATUS "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")


include_directories( /opt/rocm/opencl/include/ /opt/rocm/hip/include /opt/rocm/rocfft/include /usr/include/mpich /opt/rocm/opencl/include/ /usr/local/cuda/include /opt/rocm/llvm/bin)
link_directories( /opt/rocm/hip/include /usr/lib/x86_64-linux-gnu /opt/rocm/lib /usr/local/cuda/lib64/ /opt/rocm/llvm/bin)

add_executable(mwe-hipsycl-fft ${MY_SOURCE_FILES})

# set_target_properties(mwe-hipsycl-fft PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mwe-hipsycl-fft PUBLIC clFFT rocfft hipfft)
# target_include_directories(mwe-hipsycl-fft PUBLIC CUDA::cufft)
add_sycl_to_target(TARGET mwe-hipsycl-fft)
