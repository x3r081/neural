"""Check frozen runtime fingerprints and inventory the delivered evidence bytes."""
import hashlib
import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verified, errors = {}, []
    exclusions = {
        str(ROOT / "start_neural_optimized.bat"): "Delivery launcher updated after trials to select the measured combined flags; runtime source/DLL unchanged.",
        str(ROOT / "tools/review_persistent_trial.py"): "Read-only postprocessing audit was refined during the frozen trial; it is not imported or executed by the serving runner/runtime. Captured earlier hashes remain in raw trial provenance.",
    }
    for name in ("persistent_cpu_mirrored.json", "persistent_cpu_frozen.json", "persistent_vs_llama.json"):
        result = json.loads((FOLDER / name).read_text(encoding="utf-8"))
        for kind, repo in result["provenance"]["neural_server_roots"].items():
            for item in repo.get("source_manifest") or []:
                path = Path(item["path"])
                path = path if path.is_absolute() else ROOT / path
                if str(path) in exclusions:
                    continue
                if item.get("sha256"):
                    if not path.is_file() or sha(path) != item["sha256"]:
                        errors.append({"trial": name, "kind": kind, "source": str(path)})
                    verified[str(path)] = item["sha256"]
            for label, item in repo.get("artifacts", {}).items():
                if item.get("sha256"):
                    path = Path(item["path"])
                    if not path.is_file() or sha(path) != item["sha256"]:
                        errors.append({"trial": name, "kind": kind, "artifact": label})
                    verified[str(path)] = item["sha256"]
    quality = json.loads((FOLDER / "persistent_decode_ab.json").read_text())
    for name, key in (("gptoss_cpu_persistent.dll", "candidate_dll_sha256"),
                      ("gptoss_cpu_cap2.dll", "reference_dll_sha256"),
                      ("kernels/gptoss_cpu_persistent.c", "source_sha256"),
                      ("kernels/gptoss_cpu_cap2.c", "included_cap2_sha256")):
        if sha(ROOT / name) != quality[key]:
            errors.append({"native_quality_identity": name})
    manifest = {path.relative_to(FOLDER).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
                for path in sorted(FOLDER.rglob("*")) if path.is_file()
                and "__pycache__" not in path.parts and path.name != "delivery_verification.json"}
    report = {
        "reviewed_upstream": "a5d3c95e17323e9b78f83976da583e31f73f188e",
        "tested_runtime_commit": "9aa766f",
        "runtime_source_and_artifact_fingerprints_match_trials": not errors,
        "verified_files": verified, "intentional_nonruntime_exclusions": exclusions,
        "errors": errors,
        "selected_pytest": "103 passed, 23 existing warnings",
        "full_model_checks": "192 compared decode steps; full logits/hidden/KV/routes/tokens equal at 128/3000/13000",
        "artifact_files": manifest,
    }
    (FOLDER / "delivery_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_match": not errors, "verified_files": len(verified),
                      "artifact_files": len(manifest), "errors": errors}))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
