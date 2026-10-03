"""One-time setup: build the expert store from the downloaded checkpoint.

The store holds each expert's source MXFP4 codes and E8M0 scales byte-exactly (the only
transform is de-interleaving gate/up rows), one file per layer (~1.6 GiB each, ~57 GiB total),
plus expert_biases.pt. Streaming, bounded RAM, no GPU. Takes a few minutes on NVMe.

    python tools/build_store.py [--model-dir DIR] [--store-dir DIR]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as NP                                                   # noqa: E402
from neural.moe.gptoss_adapter import build_gptoss_prepacked_store   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=NP.MODEL_DIR, help="Hugging Face checkpoint openai/gpt-oss-120b")
ap.add_argument("--store-dir", default=NP.STORE_DIR, help="where to write the store (~57 GiB, fast disk)")
A = ap.parse_args()
if os.path.exists(os.path.join(A.store_dir, "expert_biases.pt")):
    sys.exit(f"{A.store_dir} already contains a store; refusing to overwrite it")
t0 = time.time()
meta = build_gptoss_prepacked_store(A.model_dir, A.store_dir,
                                    progress=lambda li: print(f"layer {li} done ({time.time() - t0:.0f} s)", flush=True))
print(json.dumps(meta, indent=1, default=str))
