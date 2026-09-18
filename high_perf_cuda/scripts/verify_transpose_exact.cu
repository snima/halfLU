#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cassert>
#include "../include/cuda_utils.cuh"

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_submatrix_kernel_coalesced(
    const __half* __restrict__ idata,
    int ldi,
    __half* __restrict__ odata,
    int ldo,
    int width,  // cols in idata
    int height  // rows in idata
) {
    __shared__ __half tile[TILE_DIM][TILE_DIM + 1];

    int row_in = blockIdx.y * TILE_DIM + threadIdx.x;
    int col_in = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (row_in < height && (col_in + j) < width) {
            tile[threadIdx.y + j][threadIdx.x] = idata[static_cast<size_t>(col_in + j) * ldi + row_in];
        }
    }
    __syncthreads();

    int row_out = blockIdx.x * TILE_DIM + threadIdx.x;
    int col_out = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (row_out < width && (col_out + j) < height) {
            odata[static_cast<size_t>(col_out + j) * ldo + row_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    int widths[] = {37, 256, 512, 1024, 2048, 4096};
    int heights[] = {53, 512, 1024, 2048, 4096, 8192};
    int num_tests = 6;

    for (int t = 0; t < num_tests; ++t) {
        int w = widths[t];
        int h = heights[t];
        int ldi = h + 17; // non-trivial stride
        int ldo = w + 23;

        std::vector<__half> h_in(static_cast<size_t>(w) * ldi);
        for (int c = 0; c < w; ++c) {
            for (int r = 0; r < h; ++r) {
                // Use distinct 16-bit patterns
                uint16_t pat = static_cast<uint16_t>((c * 37 + r * 19 + 1) & 0x7FFF);
                h_in[c * ldi + r] = *reinterpret_cast<__half*>(&pat);
            }
        }

        __half *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(h) * ldo * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(__half), cudaMemcpyHostToDevice));

        dim3 block(TILE_DIM, BLOCK_ROWS);
        dim3 grid((w + TILE_DIM - 1) / TILE_DIM, (h + TILE_DIM - 1) / TILE_DIM);

        transpose_submatrix_kernel_coalesced<<<grid, block>>>(d_in, ldi, d_out, ldo, w, h);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__half> h_out(static_cast<size_t>(h) * ldo);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__half), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        for (int c = 0; c < w; ++c) {
            for (int r = 0; r < h; ++r) {
                uint16_t expected = *reinterpret_cast<uint16_t*>(&h_in[c * ldi + r]);
                uint16_t actual = *reinterpret_cast<uint16_t*>(&h_out[r * ldo + c]);
                if (actual != expected) {
                    mismatches++;
                    if (mismatches <= 5) {
                        std::cerr << "Mismatch at (c=" << c << ", r=" << r << "): expected " << expected << ", got " << actual << std::endl;
                    }
                }
            }
        }

        if (mismatches == 0) {
            std::cout << "[PASS] Test " << t << ": w=" << w << ", h=" << h << " (0 bit mismatches)" << std::endl;
        } else {
            std::cerr << "[FAIL] Test " << t << ": w=" << w << ", h=" << h << " (" << mismatches << " mismatches)" << std::endl;
            return 1;
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }
    std::cout << "ALL 6 RECTANGULAR & STRIDED COALESCED TRANSPOSE TESTS PASSED BIT-EXACT!" << std::endl;
    return 0;
}
