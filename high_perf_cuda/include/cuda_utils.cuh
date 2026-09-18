#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression + ": " +
            cudaGetErrorString(status));
    }
}

inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression +
            ": cuBLAS error code " + std::to_string(status));
    }
}

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    float stop_and_sync(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// Monotone bit-key packing for 64-bit atomic/reduction argmax.
__host__ __device__ __forceinline__ unsigned short absolute_bits(__half value) {
#if defined(__CUDA_ARCH__)
    return __half_as_ushort(value) & 0x7fffu;
#else
    unsigned short raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw & 0x7fffu;
#endif
}

__host__ __device__ __forceinline__ bool bits_finite(unsigned short bits) {
    return bits < 0x7c00u;
}

__host__ __device__ __forceinline__ unsigned long long pack_key(unsigned short abs_bits, int index) {
    return (static_cast<unsigned long long>(abs_bits) << 32) | static_cast<unsigned int>(~index);
}

__host__ __device__ __forceinline__ int key_index(unsigned long long key) {
    return static_cast<int>(~static_cast<unsigned int>(key & 0xffffffffull));
}

__host__ __device__ __forceinline__ float key_magnitude(unsigned long long key) {
    const unsigned short bits = static_cast<unsigned short>((key >> 32) & 0xffffull);
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(bits));
#else
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
#endif
}

__host__ __device__ __forceinline__ bool key_empty(unsigned long long key) {
    return (key >> 32) == 0ull;
}

__device__ __forceinline__ float separation_ratio_device(unsigned long long first, unsigned long long second) {
    const float maximum = key_empty(first) ? -1.0f : key_magnitude(first);
    const float runner_up = key_empty(second) ? 0.0f : key_magnitude(second);
    if (runner_up == 0.0f) return maximum > 0.0f ? 1e30f : 0.0f;
    return maximum / runner_up;
}
