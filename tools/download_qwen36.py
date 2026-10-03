"""Fetch the pinned, unmodified official checkpoint; never quantize weights."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destination", default=r"F:\Models\Qwen3.6-35B-A3B")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    dst = Path(args.destination).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    files = [s for s in info.siblings if s.rfilename.endswith(
        (".safetensors", ".json", ".jinja", ".md")) or s.rfilename == "LICENSE"]
    required = sum(s.size or 0 for s in files)
    existing = sum((dst / s.rfilename).stat().st_size for s in files
                   if (dst / s.rfilename).is_file())
    if shutil.disk_usage(dst).free < max(0, required-existing) + 5*2**30:
        raise RuntimeError("Insufficient free space for pinned checkpoint plus reserve")
    record = {"repo_id": REPO, "revision": REVISION,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "expected_bytes": required, "status": "downloading",
              "files": [{"name": s.rfilename, "bytes": s.size,
                         "sha256": s.lfs.sha256 if s.lfs else None} for s in files]}
    receipt = dst / "neural_download_receipt.json"
    receipt.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("repo_id", "revision", "expected_bytes", "status")}), flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=str(dst),
                      allow_patterns=[s.rfilename for s in files], max_workers=args.workers)
    for s in files:
        if not (dst/s.rfilename).is_file() or (dst/s.rfilename).stat().st_size != s.size:
            raise RuntimeError(f"Missing or wrong-size file: {s.rfilename}")
    record.update(status="downloaded_size_verified",
                  finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print("DOWNLOAD_COMPLETE: all pinned files present with expected sizes", flush=True)


if __name__ == "__main__":
    main()
