# Final V100 Pivoting Evidence

- GPU: Tesla V100-PCIE-32GB (GPU 0)
- CUDA: 11.4
- Source SHA-256: `3b932cc3859e0164c78d4f83290a3c30ed9d48c8b5e628e0d3b545b09dc3a81f`
- Binary SHA-256: `ee9d4ae937306ae62052af77b6e6bcd0ea2f80a68afbc2ec72802d72f75fdab9`
- Completed runs: 1300/1300
- Arithmetic: FP16 storage, separate FP16 multiplication and subtraction
- Scaling: disabled in the pivoting comparison

## Evidence Contracts

| Tier | Raw runs | Repetitions | Warm-ups | Orders | Methods |
|---|---:|---:|---:|---|---|
| replicated | 1260 | 3 | 1 | 128, 256, 512, 1,024, 2,048 | PP, DP, GP, ScaP, RP, CP, ScPP |
| frontier_4k_10k | 18 | 1 | 1 | 4,096, 8,192, 10,240 | PP, DP, GP, ScaP, RP, ScPP |
| frontier_20k | 6 | 1 | 1 | 20,480 | PP, DP, GP, ScaP, RP, ScPP |
| frontier_30k_40k | 10 | 1 | 0 | 30,720, 40,960 | PP, DP, GP, ScaP, RP |
| frontier_50k_60k | 6 | 1 | 0 | 51,200, 61,440 | PP, DP, GP |

## Blocked Frontier

| n | b | Method | Runs | Median time (s) | Overhead vs PP | Median reconstruction | Max multiplier | Tier |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 128 | 64 | PP | 18 | 0.009 | +0.0% | 8.090e-03 | 1 | replicated |
| 128 | 64 | DP | 18 | 0.009 | +0.8% | 9.531e-03 | 0.990234 | replicated |
| 128 | 64 | GP | 18 | 0.009 | +2.0% | 8.340e-03 | 1.00977 | replicated |
| 128 | 64 | ScaP | 18 | 0.011 | +28.7% | 7.718e-03 | 1 | replicated |
| 128 | 64 | RP | 18 | 0.012 | +34.8% | 6.846e-03 | 1 | replicated |
| 128 | 64 | CP | 18 | 0.012 | +35.5% | 6.084e-03 | 1 | replicated |
| 128 | 64 | ScPP | 18 | 0.009 | +0.7% | 8.429e-03 | 1.81641 | replicated |
| 256 | 64 | PP | 18 | 0.019 | +0.0% | 1.498e-02 | 1 | replicated |
| 256 | 64 | DP | 18 | 0.019 | +2.1% | 1.741e-02 | 0.999023 | replicated |
| 256 | 64 | GP | 18 | 0.019 | +1.9% | 1.414e-02 | 1.00977 | replicated |
| 256 | 64 | ScaP | 18 | 0.024 | +25.7% | 1.431e-02 | 1 | replicated |
| 256 | 64 | RP | 18 | 0.024 | +26.2% | 1.390e-02 | 1 | replicated |
| 256 | 64 | CP | 18 | 0.036 | +89.4% | 1.181e-02 | 1 | replicated |
| 256 | 64 | ScPP | 18 | 0.019 | -0.0% | 1.665e-02 | 1.39746 | replicated |
| 512 | 64 | PP | 18 | 0.040 | +0.0% | 1.881e-02 | 1 | replicated |
| 512 | 64 | DP | 18 | 0.040 | +1.7% | 1.872e-02 | 1 | replicated |
| 512 | 64 | GP | 18 | 0.040 | +2.1% | 1.679e-02 | 1.00977 | replicated |
| 512 | 64 | ScaP | 18 | 0.050 | +25.3% | 1.623e-02 | 1 | replicated |
| 512 | 64 | RP | 18 | 0.048 | +23.0% | 1.563e-02 | 1 | replicated |
| 512 | 64 | CP | 18 | 0.155 | +290.7% | 1.378e-02 | 1 | replicated |
| 512 | 64 | ScPP | 18 | 0.039 | -0.1% | 1.696e-02 | 1.46289 | replicated |
| 1,024 | 64 | PP | 18 | 0.084 | +0.0% | 3.361e-02 | 1 | replicated |
| 1,024 | 64 | DP | 18 | 0.086 | +3.1% | 3.388e-02 | 1 | replicated |
| 1,024 | 64 | GP | 18 | 0.085 | +2.4% | 3.243e-02 | 1.00977 | replicated |
| 1,024 | 64 | ScaP | 18 | 0.105 | +25.6% | 2.944e-02 | 1 | replicated |
| 1,024 | 64 | RP | 18 | 0.100 | +20.5% | 2.939e-02 | 1 | replicated |
| 1,024 | 64 | CP | 18 | 0.880 | +953.2% | 2.767e-02 | 1 | replicated |
| 1,024 | 64 | ScPP | 18 | 0.083 | -0.2% | 3.305e-02 | 1.42969 | replicated |
| 2,048 | 64 | PP | 18 | 0.187 | +0.0% | 6.146e-02 | 1 | replicated |
| 2,048 | 64 | DP | 18 | 0.193 | +3.0% | 5.536e-02 | 1 | replicated |
| 2,048 | 64 | GP | 18 | 0.196 | +2.2% | 5.699e-02 | 1.00977 | replicated |
| 2,048 | 64 | ScaP | 18 | 0.235 | +25.3% | 5.416e-02 | 1 | replicated |
| 2,048 | 64 | RP | 18 | 0.216 | +15.3% | 5.278e-02 | 1 | replicated |
| 2,048 | 64 | CP | 18 | 5.898 | +3017.8% | 3.965e-02 | 1 | replicated |
| 2,048 | 64 | ScPP | 18 | 0.187 | -0.7% | 6.506e-02 | 1.47656 | replicated |
| 4,096 | 64 | PP | 1 | 0.515 | +0.0% | 1.199e-01 | 1 | frontier_4k_10k |
| 4,096 | 64 | DP | 1 | 0.536 | +4.1% | 1.219e-01 | 0.999512 | frontier_4k_10k |
| 4,096 | 64 | GP | 1 | 0.533 | +3.5% | 1.125e-01 | 1.00977 | frontier_4k_10k |
| 4,096 | 64 | ScaP | 1 | 0.637 | +23.7% | 1.142e-01 | 1 | frontier_4k_10k |
| 4,096 | 64 | RP | 1 | 0.575 | +11.6% | 1.064e-01 | 1 | frontier_4k_10k |
| 4,096 | 64 | ScPP | 1 | 0.515 | +0.1% | 1.082e-01 | 1.00098 | frontier_4k_10k |
| 8,192 | 128 | PP | 1 | 1.982 | +0.0% | 2.481e-01 | 1 | frontier_4k_10k |
| 8,192 | 128 | DP | 1 | 2.054 | +3.6% | 1.979e-01 | 1 | frontier_4k_10k |
| 8,192 | 128 | GP | 1 | 2.026 | +2.3% | 1.882e-01 | 1.00977 | frontier_4k_10k |
| 8,192 | 128 | ScaP | 1 | 2.330 | +17.6% | 2.426e-01 | 1 | frontier_4k_10k |
| 8,192 | 128 | RP | 1 | 2.134 | +7.7% | 2.566e-01 | 1 | frontier_4k_10k |
| 8,192 | 128 | ScPP | 1 | 1.992 | +0.5% | 1.800e-01 | 1 | frontier_4k_10k |
| 10,240 | 128 | PP | 1 | 3.339 | +0.0% | 3.340e-01 | 1 | frontier_4k_10k |
| 10,240 | 128 | DP | 1 | 3.451 | +3.4% | 3.726e-01 | 1 | frontier_4k_10k |
| 10,240 | 128 | GP | 1 | 3.401 | +1.9% | 2.583e-01 | 1.00977 | frontier_4k_10k |
| 10,240 | 128 | ScaP | 1 | 3.807 | +14.0% | 2.180e-01 | 1 | frontier_4k_10k |
| 10,240 | 128 | RP | 1 | 3.511 | +5.2% | 2.122e-01 | 1 | frontier_4k_10k |
| 10,240 | 128 | ScPP | 1 | 3.351 | +0.4% | 2.855e-01 | 1 | frontier_4k_10k |
| 20,480 | 512 | PP | 1 | 22.308 | +0.0% | 5.994e-01 | 1 | frontier_20k |
| 20,480 | 512 | DP | 1 | 22.750 | +2.0% | 5.389e-01 | 1 | frontier_20k |
| 20,480 | 512 | GP | 1 | 22.514 | +0.9% | 5.124e-01 | 1.00977 | frontier_20k |
| 20,480 | 512 | ScaP | 1 | 23.971 | +7.5% | 4.545e-01 | 1 | frontier_20k |
| 20,480 | 512 | RP | 1 | 23.044 | +3.3% | 4.502e-01 | 1 | frontier_20k |
| 20,480 | 512 | ScPP | 1 | 22.342 | +0.2% | 5.994e-01 | 1 | frontier_20k |
| 30,720 | 1024 | PP | 1 | 75.466 | +0.0% | 6.772e-01 | 1 | frontier_30k_40k |
| 30,720 | 1024 | DP | 1 | 76.493 | +1.4% | 7.647e-01 | 1 | frontier_30k_40k |
| 30,720 | 1024 | GP | 1 | 75.972 | +0.7% | 8.901e-01 | 1.00977 | frontier_30k_40k |
| 30,720 | 1024 | ScaP | 1 | 78.837 | +4.5% | 7.468e-01 | 1 | frontier_30k_40k |
| 30,720 | 1024 | RP | 1 | 77.019 | +2.1% | 9.296e-01 | 1 | frontier_30k_40k |
| 40,960 | 1024 | PP | 1 | 163.957 | +0.0% | 1.155e+00 | 1 | frontier_30k_40k |
| 40,960 | 1024 | DP | 1 | 165.371 | +0.9% | 1.308e+00 | 1 | frontier_30k_40k |
| 40,960 | 1024 | GP | 1 | 164.777 | +0.5% | 9.899e-01 | 1.00977 | frontier_30k_40k |
| 40,960 | 1024 | ScaP | 1 | 169.390 | +3.3% | 1.150e+00 | 1 | frontier_30k_40k |
| 40,960 | 1024 | RP | 1 | 166.010 | +1.3% | 1.098e+00 | 1 | frontier_30k_40k |
| 51,200 | 1024 | PP | 1 | 301.415 | +0.0% | 1.269e+00 | 1 | frontier_50k_60k |
| 51,200 | 1024 | DP | 1 | 303.789 | +0.8% | 1.308e+00 | 1 | frontier_50k_60k |
| 51,200 | 1024 | GP | 1 | 302.785 | +0.5% | 2.388e+00 | 1.00977 | frontier_50k_60k |
| 61,440 | 1024 | PP | 1 | 501.911 | +0.0% | 2.069e+00 | 1 | frontier_50k_60k |
| 61,440 | 1024 | DP | 1 | 505.658 | +0.7% | 1.812e+00 | 1 | frontier_50k_60k |
| 61,440 | 1024 | GP | 1 | 503.975 | +0.4% | 3.162e+00 | 1.00977 | frontier_50k_60k |

## Method Coverage

| Method | Largest tested blocked order |
|---|---:|
| PP | 61,440 |
| DP | 61,440 |
| GP | 61,440 |
| ScaP | 40,960 |
| RP | 40,960 |
| CP | 2,048 |
| ScPP | 20,480 |

The replicated grid supports runtime comparisons through n=2,048. The larger cells are progressively sampled frontier tests and must not be presented as equally replicated confidence evidence. Panel-local methods are algorithmic variants of the global pivoting rules.
