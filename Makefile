# ==============================================================================
# halfLU — Top-Level Automation Makefile
# ACM Transactions on Mathematical Software (TOMS)
# ==============================================================================

NVCC        ?= nvcc
CUDA_ARCH   ?= sm_70
NVCC_FLAGS  ?= -O3 -arch=$(CUDA_ARCH) --std=c++17 -lcublas
BIN_DIR     ?= bin

# Binaries
BIN_PREDICTIVE   = $(BIN_DIR)/blocked_predictive_validation
BIN_NLA_EXT      = $(BIN_DIR)/nla_extension_validation
BIN_PIVOT_FROZEN = $(BIN_DIR)/pivoting_runtime_validation
BIN_PIVOT_FUSED  = $(BIN_DIR)/pivoting_search_cost_validation

ALL_BINS = $(BIN_PREDICTIVE) $(BIN_NLA_EXT) $(BIN_PIVOT_FROZEN) $(BIN_PIVOT_FUSED)

.PHONY: all clean quick-reproduce kernels high-perf help

help:
	@echo "halfLU Build and Reproduction Targets:"
	@echo "  make quick-reproduce    Run fast 30-second CPU-only reproduction of all figures/tables"
	@echo "  make kernels            Compile standalone CUDA benchmark validators (requires nvcc)"
	@echo "  make high-perf          Build 38 TFLOPS Tensor Core benchmark suite (high_perf_cuda)"
	@echo "  make clean              Remove compiled binaries and temporary artifacts"
	@echo ""
	@echo "Supported CUDA architectures (default sm_70 for Tesla V100):"
	@echo "  make kernels CUDA_ARCH=sm_70    # NVIDIA Tesla V100 (Volta)"
	@echo "  make kernels CUDA_ARCH=sm_80    # NVIDIA A100 (Ampere)"
	@echo "  make kernels CUDA_ARCH=sm_86    # NVIDIA RTX 3080/3090 (Ampere)"
	@echo "  make kernels CUDA_ARCH=sm_90    # NVIDIA H100 (Hopper)"

quick-reproduce:
	@bash scripts/run_all_figures_and_tables.sh

high-perf:
	@mkdir -p high_perf_cuda/build && cd high_perf_cuda/build && cmake .. && make -j$$(nproc)
	@echo "Built high_perf_cuda/build/high_perf_lu_benchmark successfully."

kernels: $(ALL_BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_PREDICTIVE): cuda/dynamic_scaling_validation.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_NLA_EXT): cuda/dynamic_scaling_validation_nla.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_PIVOT_FROZEN): cuda/pivoting_runtime_validation.cu | $(BIN_DIR)
	$(NVCC) -O3 -arch=$(CUDA_ARCH) --std=c++17 -o $@ $<

$(BIN_PIVOT_FUSED): cuda/pivoting_search_cost_validation.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -rf $(BIN_DIR)
