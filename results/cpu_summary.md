# CPU benchmark comparison

cpu_benchmark_v1; mean ± standard deviation; r=3. Latest complete valid run per model/thread; pp and tg from the same file.

| Model | Engine | Threads | pp128 (t/s) | tg128 (t/s) | tg / Qwen |
|---|---|---:|---:|---:|---:|
| Qwen3 Q4_K_M | llama.cpp | 1 | 52.14 ± 0.01 | 20.52 ± 0.03 | 1.00x |
| Qwen3 Q4_K_M | llama.cpp | 2 | 94.37 ± 0.49 | 33.93 ± 0.21 | 1.00x |
| Qwen3 Q4_K_M | llama.cpp | 4 | 180.71 ± 0.03 | 56.19 ± 0.60 | 1.00x |
| Bonsai Q2_0_g64 | llama.cpp | 1 | 17.56 ± 0.02 | 13.57 ± 0.03 | 0.66x |
| Bonsai Q2_0_g64 | llama.cpp | 2 | 33.75 ± 0.11 | 23.23 ± 0.14 | 0.68x |
| Bonsai Q2_0_g64 | llama.cpp | 4 | 67.33 ± 0.20 | 42.84 ± 0.08 | 0.76x |
| Bonsai PQ2_0 | Prism | 1 | 41.75 ± 0.00 | 15.17 ± 0.10 | 0.74x |
| Bonsai PQ2_0 | Prism | 2 | 79.93 ± 0.34 | 27.68 ± 0.10 | 0.82x |
| Bonsai PQ2_0 | Prism | 4 | 153.76 ± 0.16 | 50.84 ± 0.27 | 0.90x |
| BitNet I2_S | BitNet | 1 | 85.33 ± 0.03 | 9.89 ± 0.02 | 0.48x |
| BitNet I2_S | BitNet | 2 | 150.21 ± 1.21 | 18.32 ± 0.04 | 0.54x |
| BitNet I2_S | BitNet | 4 | 282.74 ± 0.08 | 32.81 ± 0.20 | 0.58x |

Source files:

- Qwen3 Q4_K_M, t=1: [JSON](qwen3-q4_cpu_bench_t1_20261116T005140Z_cVn0VJ.json) / [log](../logs/qwen3-q4_cpu_bench_t1_20261116T005140Z_cVn0VJ.log)
- Qwen3 Q4_K_M, t=2: [JSON](qwen3-q4_cpu_bench_t2_20261116T005047Z_y3Tb0F.json) / [log](../logs/qwen3-q4_cpu_bench_t2_20261116T005047Z_y3Tb0F.log)
- Qwen3 Q4_K_M, t=4: [JSON](qwen3-q4_cpu_bench_t4_20261116T004014Z_soDpc4.json) / [log](../logs/qwen3-q4_cpu_bench_t4_20261116T004014Z_soDpc4.log)
- Bonsai Q2_0_g64, t=1: [JSON](bonsai-q2g64_cpu_bench_t1_20261116T105336Z_8ZuV9l.json) / [log](../logs/bonsai-q2g64_cpu_bench_t1_20261116T105336Z_8ZuV9l.log)
- Bonsai Q2_0_g64, t=2: [JSON](bonsai-q2g64_cpu_bench_t2_20261116T105435Z_qcPOrP.json) / [log](../logs/bonsai-q2g64_cpu_bench_t2_20261116T105435Z_qcPOrP.log)
- Bonsai Q2_0_g64, t=4: [JSON](bonsai-q2g64_cpu_bench_t4_20261116T105508Z_YJLKjh.json) / [log](../logs/bonsai-q2g64_cpu_bench_t4_20261116T105508Z_YJLKjh.log)
- Bonsai PQ2_0, t=1: [JSON](bonsai-pq2_prism_cpu_bench_t1_20261116T115726Z_4eIso1.json) / [log](../logs/bonsai-pq2_prism_cpu_bench_t1_20261116T115726Z_4eIso1.log)
- Bonsai PQ2_0, t=2: [JSON](bonsai-pq2_prism_cpu_bench_t2_20261116T115805Z_a49p5S.json) / [log](../logs/bonsai-pq2_prism_cpu_bench_t2_20261116T115805Z_a49p5S.log)
- Bonsai PQ2_0, t=4: [JSON](bonsai-pq2_prism_cpu_bench_t4_20261116T115827Z_O5Rqz2.json) / [log](../logs/bonsai-pq2_prism_cpu_bench_t4_20261116T115827Z_O5Rqz2.log)
- BitNet I2_S, t=1: [JSON](bitnet-i2s_cpu_bench_t1_20261116T175028Z_vjkjx1.json) / [log](../logs/bitnet-i2s_cpu_bench_t1_20261116T175028Z_vjkjx1.log)
- BitNet I2_S, t=2: [JSON](bitnet-i2s_cpu_bench_t2_20261116T175116Z_3VPX2d.json) / [log](../logs/bitnet-i2s_cpu_bench_t2_20261116T175116Z_3VPX2d.log)
- BitNet I2_S, t=4: [JSON](bitnet-i2s_cpu_bench_t4_20261116T175144Z_mWSfVL.json) / [log](../logs/bitnet-i2s_cpu_bench_t4_20261116T175144Z_mWSfVL.log)
