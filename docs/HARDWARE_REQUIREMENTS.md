# Hardware and Software Requirements

This document specifies the target hardware environments, verified configurations, and software toolchains required to evaluate and reproduce results in the **halfLU** repository.

---

## 1. Operating Modes and Hardware Matrix

| Evaluation Mode | Required Hardware | Minimum Host RAM | GPU Memory | Estimated Wall Time |
|---|---|---|---|---|
| **CPU Fast Track** | Standard x86_64 CPU (Intel / AMD, $\ge 2$ cores) | 4 GB | None (No GPU required) | $< 30$ seconds |
| **GPU Pilot Runs** ($n \le 4{,}096$) | NVIDIA GPU with Tensor Cores (sm_70, sm_75, sm_80, sm_86, sm_89, sm_90) | 16 GB | $\ge 4$ GB VRAM | $\approx 2\text{--}5$ minutes |
| **Full GPU Benchmark** ($n \le 61{,}440$) | Dedicated NVIDIA Tesla V100-PCIE-32GB or NVIDIA A100-PCIE-40/80GB | 64 GB | $\ge 32$ GB VRAM | $\approx 2\text{--}4$ hours |

---

## 2. Reference GPU Hardware (Paper Benchmark System)

The published GPU benchmarks in the paper were gathered on a dedicated HPC node with the following reference configuration:

- **GPU:** NVIDIA Tesla V100-PCIE-32GB
  - Architecture: Volta (`sm_70`)
  - Streaming Multiprocessors (SMs): 80
  - FP16 Tensor Core Peak Throughput: 125 TFLOPS
  - High Bandwidth Memory: 32 GB HBM2 @ 900 GB/s
  - Interconnect: PCIe Gen3 x16
- **Host System:**
  - CPU: Intel Xeon Silver 4210 @ 2.20GHz (10 cores / 20 threads)
  - System Memory: 128 GB DDR4 ECC RAM
  - Storage: NVMe SSD scratch space
- **Operating System:** Ubuntu 22.04 LTS (Kernel 5.15.0 x86_64)

---

## 3. Software Dependencies and Toolchains

### NVIDIA CUDA Toolchain
- **CUDA Toolkit:** $\ge 11.2$ (validated on CUDA 11.8 and CUDA 12.2)
- **Compiler:** `nvcc` with C++17 support (`-O3 -arch=sm_70 --std=c++17`)
- **GPU Driver:** NVIDIA Display Driver $\ge 470.57.02$ (CUDA 11.x) or $\ge 525.60.13$ (CUDA 12.x)
- **CUDA Libraries:** `cuBLAS` (included with the standard CUDA Toolkit installation)

### Python Environment
- **Python:** Version $\ge 3.8$ (Python 3.10 / 3.11 / 3.12 recommended)
- **Core Scientific Libraries:**
  - `numpy >= 1.22.0`
  - `pandas >= 1.4.0`
  - `matplotlib >= 3.5.0`
  - `scipy >= 1.8.0`

---

## 4. CUDA Architecture Compilation Flags

When compiling kernels with `make kernels`, select the appropriate `CUDA_ARCH` flag for your system:

```bash
# Volta (Tesla V100, Titan V)
make kernels CUDA_ARCH=sm_70

# Turing (Quadro T1000, RTX 2080 Ti, Titan RTX)
make kernels CUDA_ARCH=sm_75

# Ampere Datacenter (A100, A30, A40)
make kernels CUDA_ARCH=sm_80

# Ampere Consumer (RTX 3080, RTX 3090, A5000, A6000)
make kernels CUDA_ARCH=sm_86

# Ada Lovelace (RTX 4090, L40)
make kernels CUDA_ARCH=sm_89

# Hopper (H100 PCIe, H100 SXM5)
make kernels CUDA_ARCH=sm_90
```

---

## 5. Memory Capacity and Large-Scale Factorization Notes

At $n = 61{,}440$, an FP16 matrix occupies:
$$61{,}440 \times 61{,}440 \times 2\text{ bytes} \approx 7.55\text{ GB}$$
Full validation requires device allocations for the working matrix, pivot vectors, and auxiliary residual estimators. A GPU with at least **16 GB VRAM** is recommended for $n \le 40{,}960$, and **32 GB VRAM** for the maximum dimension $n = 61{,}440$.

For hosts with limited physical RAM, host-resident staging buffers during campaign data dumps require $\approx 16\text{--}32\text{ GB}$ available host RAM.
