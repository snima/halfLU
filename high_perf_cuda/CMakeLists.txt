cmake_minimum_required(VERSION 3.18)
project(HighPerfPivotingCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "75;70" CACHE STRING "CUDA Architectures" FORCE)

message(STATUS "Configuring High-Performance CUDA FP16 LU Benchmark")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --use_fast_math -rdc=true")

add_executable(high_perf_lu_benchmark
    src/main_benchmark.cu
)

target_link_libraries(high_perf_lu_benchmark PRIVATE
    CUDA::cublas
    CUDA::cudart
)

set_target_properties(high_perf_lu_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
