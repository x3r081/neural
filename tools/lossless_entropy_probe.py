"""Bounded, read-only lossless-compression gate for the existing expert store.

Sample one seeded random expert from each of 12 layer strata by default. Never
map/read the whole model, rewrite a store, unpack/requantize weights, or load a
GPU library. Process one expert at a time; histogram chunks are capped at 1 MiB
and source rows at 16 MiB, keeping data buffers below 160 MiB.

  project-python tools/lossless_entropy_probe.py --store <packed-store> --output <json>

Byte/nibble Shannon estimates are CALCULATED zero-order memoryless coding costs
from the sampled histogram, not universal lower bounds or whole-model results.
Raw DEFLATE level-1 ratios, timings and throughput are MEASURED CPU-only codec
results (one worker, one codec thread), not inference performance. Decoding is
verified by exact byte equality AND SHA-256 outside the timing interval. Compression
receives the original stored bytes, including the already packed scale region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import zlib

# No BLAS or OpenMP work is needed by the histograms. Pin their process defaults
# before importing numpy so this standalone probe cannot create an implicit team.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import paths as NP
from neural.moe.layout import GPTOSS_LAYOUT, GPTOSS_LAYOUT_PS4, prepacked_descriptor

MAX_ROW_BYTES = 16 << 20
MAX_HIST_CHUNK = 1 << 20


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entropy(hist):
    """Empirical H0 in bits/symbol; the histogram counts themselves are measured."""
    h = np.asarray(hist, dtype=np.float64)
    total = float(h.sum())
    if not total:
        return 0.0
    p = h[h > 0] / total
    return float(-(p * np.log2(p)).sum())


def histograms(data, *, nibbles):
    byte = np.zeros(256, np.int64)
    low = np.zeros(16, np.int64)
    high = np.zeros(16, np.int64)
    for off in range(0, len(data), MAX_HIST_CHUNK):
        a = np.frombuffer(data[off:off + MAX_HIST_CHUNK], dtype=np.uint8)
        byte += np.bincount(a, minlength=256)
        if nibbles:
            low += np.bincount(a & 15, minlength=16)
            high += np.bincount(a >> 4, minlength=16)
    return {"byte": byte, "low_nibble": low, "high_nibble": high}


def summarize_hist(hist, *, nibbles):
    nbytes = int(hist["byte"].sum())
    byte_h = entropy(hist["byte"])
    out = {"observations_label": "MEASURED", "bytes": nbytes,
           "byte_counts": hist["byte"].tolist(),
           "entropy_label": "CALCULATED", "byte_entropy_bits_per_byte": byte_h,
           "ideal_byte_memoryless_ratio": byte_h / 8.0,
           "ideal_byte_memoryless_encoded_bytes": nbytes * byte_h / 8.0}
    if nibbles:
        lo_h = entropy(hist["low_nibble"])
        hi_h = entropy(hist["high_nibble"])
        both_h = entropy(hist["low_nibble"] + hist["high_nibble"])
        out.update(low_nibble_counts=hist["low_nibble"].tolist(),
                   high_nibble_counts=hist["high_nibble"].tolist(),
                   low_nibble_entropy_bits=lo_h, high_nibble_entropy_bits=hi_h,
                   pooled_nibble_entropy_bits=both_h,
                   ideal_pooled_nibble_memoryless_ratio=both_h / 4.0,
                   ideal_separate_nibble_memoryless_ratio=(lo_h + hi_h) / 8.0,
                   high_given_low_entropy_bits=max(0.0, byte_h - lo_h))
    return out


def strata(n_layers, n_experts, count, seed):
    if not 1 <= count <= n_layers:
        raise ValueError(f"samples must be between 1 and the number of layers ({n_layers})")
    rng = random.Random(seed)
    out = []
    for i in range(count):
        lo, hi = i * n_layers // count, (i + 1) * n_layers // count
        out.append({"stratum": i, "layer_range": [lo, hi],
                    "layer": rng.randrange(lo, hi), "expert": rng.randrange(n_experts)})
    return out


def store_info(root):
    meta_path = root / "metadata.json"
    payload = meta_path.read_bytes()
    metadata = json.loads(payload)
    desc = metadata.get("layout", {})
    rep = desc.get("weight_repr")
    layouts = {"mxfp4_g32": GPTOSS_LAYOUT, "mxfp4_g32_ps4": GPTOSS_LAYOUT_PS4}
    if rep not in layouts:
        raise ValueError(f"unsupported store representation {rep!r}; require existing gpt-oss MXFP4 raw/packed store")
    layout = layouts[rep]
    expected = prepacked_descriptor(layout)
    for field in ("n_layers", "n_experts", "hidden", "gate_up_n", "down_n", "down_k", "int4_group", "slot_bytes"):
        if desc.get(field) != expected[field]:
            raise ValueError(f"metadata {field}: got {desc.get(field)!r}, expected {expected[field]!r}")
    if metadata.get("magic") != "Q80-PREPACKED-SLOTS" or metadata.get("version") != 1:
        raise ValueError("unsupported expert-store magic/version")
    if layout.slot_bytes > MAX_ROW_BYTES:
        raise ValueError("row exceeds the bounded probe's 16 MiB limit")
    return metadata, layout, sha(payload)


def regions(layout):
    gate_codes_end = layout.gate_pack_bytes
    codes_end = gate_codes_end + layout.down_pack_bytes
    gate_scales_end = codes_end + layout.gate_saz_bytes
    return {"gate_codes": (0, gate_codes_end, True),
            "down_codes": (gate_codes_end, codes_end, True),
            "gate_scales_stored": (codes_end, gate_scales_end, False),
            "down_scales_stored": (gate_scales_end, layout.slot_bytes, False)}


def compress(data, level):
    # Raw DEFLATE stream, no preprocessing or transformed/numerically changed data.
    obj = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    return obj.compress(data) + obj.flush()


def codec_probe(data, level, repeats):
    checksum = sha(data)
    # Warm the codec path once. This is CPU hot-buffer timing, not disk throughput.
    encoded = compress(data, level)
    decoded = zlib.decompress(encoded, -zlib.MAX_WBITS)
    if decoded != data or sha(decoded) != checksum:
        raise AssertionError("warm-up lossless round trip failed")
    del decoded
    encoded_sha, size = sha(encoded), len(encoded)
    compression_s, decompression_s = [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        new_encoded = compress(data, level)
        compression_s.append((time.perf_counter_ns() - t0) / 1e9)
        if len(new_encoded) != size or sha(new_encoded) != encoded_sha:
            raise AssertionError("codec produced non-deterministic compressed bytes")
        del encoded
        encoded = new_encoded
        t0 = time.perf_counter_ns()
        decoded = zlib.decompress(encoded, -zlib.MAX_WBITS)
        decompression_s.append((time.perf_counter_ns() - t0) / 1e9)
        # Every timed decode is checked; hashing/equality is excluded from codec times.
        if decoded != data or sha(decoded) != checksum:
            raise AssertionError("timed lossless round trip failed")
        del decoded
    c, d = statistics.median(compression_s), statistics.median(decompression_s)
    return {"label": "MEASURED", "codec": "zlib_raw_deflate", "level": level,
            "worker_threads": 1, "codec_internal_threads": 1, "repeats": repeats,
            "input_bytes": len(data), "compressed_bytes": size, "compressed_ratio": size / len(data),
            "input_sha256": checksum, "compressed_sha256": encoded_sha,
            "every_roundtrip_byte_equal_and_sha256_equal": True,
            "compression_s": compression_s, "decompression_s": decompression_s,
            "compression_median_s": c, "decompression_median_s": d,
            "compression_input_GBps": len(data) / c / 1e9,
            "decompression_output_GBps": len(data) / d / 1e9,
            "timing_scope": "CPU-only hot input; includes codec allocation; excludes file reads and checksums"}


def resource_snapshot():
    # Optional process accounting without introducing any worker or system probe.
    try:
        import psutil
        p = psutil.Process()
        m = p.memory_info()
        return {"rss_bytes": m.rss, "vms_bytes": m.vms, "process_threads": p.num_threads(),
                "cpu_affinity": p.cpu_affinity() if hasattr(p, "cpu_affinity") else None}
    except (ImportError, OSError, AttributeError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=Path(NP.STORE_DIR))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--level", type=int, default=1, choices=range(10))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dram-gbps", type=float, default=None,
                    help="optional separately MEASURED DRAM GB/s; enables CALCULATED sequential break-even comparison")
    args = ap.parse_args()
    if args.repeats < 1 or args.repeats > 20:
        ap.error("repeats must be between 1 and 20")
    if args.dram_gbps is not None and args.dram_gbps <= 0:
        ap.error("dram-gbps must be positive")
    root, dest = args.store.resolve(), args.output.resolve()
    if dest.is_relative_to(root):
        ap.error("the report must be outside the source store; the store is read-only")
    metadata, layout, meta_sha = store_info(root)
    sample = strata(layout.n_layers, layout.n_experts, args.samples, args.seed)
    rr = regions(layout)
    hist_total = {name: {"byte": np.zeros(256, np.int64), "low_nibble": np.zeros(16, np.int64),
                         "high_nibble": np.zeros(16, np.int64)} for name in rr}
    out = {"scope": "seeded stratified sample; not the whole model and not an inference benchmark",
           "store": str(root), "source_open_mode": "rb", "metadata_sha256": meta_sha,
           "model_id_from_metadata": metadata.get("model_id"), "store_layout": metadata["layout"],
           "scale_layout": "base byte plus packed 4-bit deltas" if layout.scale_layout_mode else "raw E8M0 bytes",
           "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
           "logical_cpus": os.cpu_count(), "numpy": np.__version__,
           "zlib_compile_version": zlib.ZLIB_VERSION, "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
           "workers": 1, "codec_internal_threads": 1, "entropy_threads": 1,
           "thread_environment": {v: os.environ[v] for v in
                                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
           "resource_start": resource_snapshot(), "resource_after_rows": [],
           "memory_bound": {"max_source_row_bytes": MAX_ROW_BYTES, "histogram_chunk_bytes": MAX_HIST_CHUNK,
                            "data_buffer_budget_bytes": 160 << 20, "one_expert_at_a_time": True},
           "sampling": {"seed": args.seed, "layer_strata": args.samples, "selection": sample,
                        "population_experts": layout.n_layers * layout.n_experts,
                        "sample_fraction": args.samples / (layout.n_layers * layout.n_experts)},
           "entropy_interpretation": "CALCULATED H0 memoryless coding estimate from sample histograms. It excludes model/header costs and is not a universal lower bound; correlations may permit a different ratio.",
           "rows": [], "all_roundtrips_verified": False}
    start = time.perf_counter()
    try:
        for selected in sample:
            layer, expert = selected["layer"], selected["expert"]
            path = root / f"layer_{layer}.slots"
            before = path.stat()
            expected_size = layout.n_experts * layout.slot_bytes
            if before.st_size != expected_size:
                raise ValueError(f"{path.name}: length {before.st_size}, expected {expected_size}")
            t0 = time.perf_counter()
            with path.open("rb") as source:
                source.seek(expert * layout.slot_bytes)
                data = source.read(layout.slot_bytes)
            read_s = time.perf_counter() - t0
            if len(data) != layout.slot_bytes:
                raise EOFError(f"short read for layer {layer}, expert {expert}")
            view = memoryview(data)
            row = {**selected, "read_s_MEASURED": read_s,
                   "source_layer_sha256_from_metadata_not_recomputed": metadata.get("sha256_per_layer", {}).get(str(layer)),
                   "entropy": {}}
            for name, (lo, hi, nibs) in rr.items():
                hist = histograms(view[lo:hi], nibbles=nibs)
                row["entropy"][name] = summarize_hist(hist, nibbles=nibs)
                for key in hist:
                    hist_total[name][key] += hist[key]
            row["codec"] = codec_probe(data, args.level, args.repeats)
            after = path.stat()
            row["source_file_size_and_mtime_unchanged"] = (before.st_size == after.st_size
                                                            and before.st_mtime_ns == after.st_mtime_ns)
            if not row["source_file_size_and_mtime_unchanged"]:
                raise RuntimeError("source file changed while the sample was being read")
            out["rows"].append(row)
            out["resource_after_rows"].append(resource_snapshot())
            print(f"L{layer} E{expert}: ratio {row['codec']['compressed_ratio']:.4f}, "
                  f"decode {row['codec']['decompression_output_GBps']:.3f} GB/s, bytes+SHA verified", flush=True)
            del view, data
        out["sample_entropy"] = {name: summarize_hist(hist_total[name], nibbles=rr[name][2]) for name in rr}
        codes_hist = {key: hist_total["gate_codes"][key] + hist_total["down_codes"][key]
                      for key in ("byte", "low_nibble", "high_nibble")}
        out["sample_entropy"]["codes_combined"] = summarize_hist(codes_hist, nibbles=True)
        codecs = [r["codec"] for r in out["rows"]]
        input_bytes = sum(r["input_bytes"] for r in codecs)
        compressed_bytes = sum(r["compressed_bytes"] for r in codecs)
        decode_s = sum(r["decompression_median_s"] for r in codecs)
        encode_s = sum(r["compression_median_s"] for r in codecs)
        out["sample_codec_summary"] = {"label": "MEASURED", "sampled_input_bytes": input_bytes,
                                       "sampled_compressed_bytes": compressed_bytes,
                                       "weighted_compressed_ratio": compressed_bytes / input_bytes,
                                       "decompression_output_GBps_from_sum_of_row_medians": input_bytes / decode_s / 1e9,
                                       "compression_input_GBps_from_sum_of_row_medians": input_bytes / encode_s / 1e9,
                                       "codec_level": args.level, "workers": 1, "not_inference_speed": True}
        if args.dram_gbps is not None:
            ratio = compressed_bytes / input_bytes
            dec_gbps = input_bytes / decode_s / 1e9
            sequential_ratio = ratio + args.dram_gbps / dec_gbps
            out["sequential_bandwidth_gate"] = {
                "label": "CALCULATED", "supplied_dram_GBps": args.dram_gbps,
                "single_thread_decode_output_GBps_MEASURED": dec_gbps,
                "compressed_read_plus_decode_over_raw_read_time": sequential_ratio,
                "necessary_decode_GBps_to_break_even": args.dram_gbps / (1 - ratio) if ratio < 1 else None,
                "break_even_on_this_simplified_sequential_model": sequential_ratio < 1,
                "assumptions": "Read compressed bytes at supplied DRAM rate, then decode at measured single-thread hot-buffer rate. Ignores output write/read bus traffic, overlap, multicore scaling and inference compute. Passing does not prove runtime gain."}
        out["all_roundtrips_verified"] = True
    except BaseException as exc:
        out["error"] = repr(exc)
        raise
    finally:
        out["wall_s_MEASURED"] = time.perf_counter() - start
        out["resource_end"] = resource_snapshot()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
