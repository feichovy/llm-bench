from pathlib import Path
import csv
import json
import math
import sys

root = Path(__file__).resolve().parents[1]
results = root / "results"
models = [
    ("Qwen3 Q4_K_M", "llama.cpp", "qwen3-q4", "Qwen3-1.7B-Q4_K_M.gguf"),
    ("Bonsai Q2_0_g64", "llama.cpp", "bonsai-q2g64", "Ternary-Bonsai-1.7B-Q2_0_g64.gguf"),
    ("Bonsai PQ2_0", "Prism", "bonsai-pq2_prism", "Ternary-Bonsai-1.7B-PQ2_0.gguf"),
    ("BitNet I2_S", "BitNet", "bitnet-i2s", "ggml-model-i2_s.gguf"),
]
rows = []
missing = []

for label, engine, prefix, model_file in models:
    for threads in (1, 2, 4):
        valid = []
        pattern = f"{prefix}_cpu_bench_t{threads}_*.json"
        for path in sorted(results.glob(pattern), reverse=True):
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, list) or len(data) != 2:
                    raise ValueError("expected two benchmark records")

                expected = dict(
                    n_threads=threads, n_gpu_layers=0, n_depth=0,
                    n_batch=2048, n_ubatch=512,
                    type_k="f16", type_v="f16", flash_attn=-1,
                )
                for item in data:
                    if Path(item["model_filename"]).name != model_file:
                        raise ValueError("model filename mismatch")
                    for key, value in expected.items():
                        if item.get(key) != value:
                            raise ValueError(
                                f"{key}: expected {value}, got {item.get(key)}"
                            )
                    if len(item["samples_ts"]) != 3:
                        raise ValueError("expected three repetitions")
                    for key in ("avg_ts", "stddev_ts"):
                        if not math.isfinite(item[key]) or item[key] < 0:
                            raise ValueError(f"invalid {key}")
                    if item["avg_ts"] == 0:
                        raise ValueError("zero speed")

                tests = {(item["n_prompt"], item["n_gen"]): item for item in data}
                if set(tests) != {(128, 0), (0, 128)}:
                    raise ValueError("expected separate pp128 and tg128")
                pp, tg = tests[(128, 0)], tests[(0, 128)]
                if any(
                    pp[key] != tg[key]
                    for key in ("build_commit", "model_filename")
                ):
                    raise ValueError("build commit or model path mismatch within file")
                valid.append((path, pp, tg))

            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"SKIP {path.name}: {error}", file=sys.stderr)

        if not valid:
            missing.append(f"{label}, t={threads}")
            continue

        path, pp, tg = valid[0]
        if len(valid) > 1:
            print(
                f"SELECT latest of {len(valid)} valid runs: {path.name}",
                file=sys.stderr,
            )
        rows.append(dict(
            model=label, engine=engine, threads=threads,
            pp_mean_tps=pp["avg_ts"], pp_std_tps=pp["stddev_ts"],
            tg_mean_tps=tg["avg_ts"], tg_std_tps=tg["stddev_ts"],
            build_commit=pp["build_commit"], source_json=path.name,
        ))

if missing:
    sys.exit("Summary not written; missing valid runs: " + "; ".join(missing))

for engine in {row["engine"] for row in rows}:
    commits = {
        row["build_commit"] for row in rows if row["engine"] == engine
    }
    if len(commits) != 1:
        sys.exit(
            f"Summary not written; mixed build commits for {engine}: {commits}"
        )

baseline = {
    row["threads"]: row["tg_mean_tps"]
    for row in rows if row["model"] == "Qwen3 Q4_K_M"
}
for row in rows:
    row["tg_vs_qwen"] = row["tg_mean_tps"] / baseline[row["threads"]]

table = [
    "| Model | Engine | Threads | pp128 (t/s) | tg128 (t/s) | tg / Qwen |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    table.append(
        f"| {row['model']} | {row['engine']} | {row['threads']} | "
        f"{row['pp_mean_tps']:.2f} ± {row['pp_std_tps']:.2f} | "
        f"{row['tg_mean_tps']:.2f} ± {row['tg_std_tps']:.2f} | "
        f"{row['tg_vs_qwen']:.2f}x |"
    )

csv_path = results / "cpu_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

sources = [
    f"- {row['model']}, t={row['threads']}: "
    f"[JSON]({row['source_json']}) / "
    f"[log](../logs/{Path(row['source_json']).stem}.log)"
    for row in rows
]
md_path = results / "cpu_summary.md"
md_path.write_text(
    "# CPU benchmark comparison\n\n"
    "cpu_benchmark_v1; mean ± standard deviation; r=3. "
    "Latest complete valid run per model/thread; "
    "pp and tg from the same file.\n\n"
    + "\n".join(table)
    + "\n\nSource files:\n\n"
    + "\n".join(sources) + "\n",
    encoding="utf-8",
)

print("\n".join(table))
print(f"\nSaved: {md_path}\nSaved: {csv_path}")
