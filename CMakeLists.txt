cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "/usr/bin/clang-14")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-14")
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 75)
# endif()

project(mwe-hipsycl-cufft VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(SRC_PATH "src" CACHE PATH "Sources.")

set(MY_SOURCE_FILES
    ${SRC_PATH}/mwe.cpp)

find_package(hipSYCL REQUIRED)
# find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(mwe-hipsycl-fft ${MY_SOURCE_FILES})

# set_target_properties(mwe-hipsycl-fft PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mwe-hipsycl-fft PUBLIC CUDA::cudart CUDA::cufft)
# target_include_directories(mwe-hipsycl-fft PUBLIC CUDA::cufft)
add_sycl_to_target(TARGET mwe-hipsycl-fft)
