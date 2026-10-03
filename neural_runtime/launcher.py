"""One-window GPT-OSS startup, with ownership of its background server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .defaults import DEFAULT_API_MODEL, DEFAULT_MODEL_LABEL

ROOT = Path(__file__).resolve().parents[1]


def prepare_launch(args):
    """Validate the novice GPT-OSS path before allocating model tensors."""
    from .__main__ import parser, model_paths, native_plan
    from .model_spec import inspect_model

    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    command = ["serve", "--mode", "fast", "--context", str(args.context),
               "--device", args.device, "--port", str(args.port), "--host", "127.0.0.1"]
    for flag, value in (("--model-dir", args.model_dir), ("--store-dir", args.store_dir)):
        if value:
            command.extend((flag, value))
    serving = parser().parse_args(command)
    model, store = model_paths(serving)
    # Fast mode refuses reference adapters, even when an old Qwen config exists.
    try:
        spec = inspect_model(model, store, mode="fast").to_dict()
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError(f"Start Neural uses {DEFAULT_MODEL_LABEL}. Run install_neural.bat or configure "
                         f"its existing model/store paths. Setup details: {exc}") from exc
    if spec["backend"] != "gptoss-native-mxfp4":
        raise ValueError("Start Neural requires GPT-OSS 120B. Other models use the advanced serve command.")
    hardware, plan = native_plan(serving, spec)
    if not plan.get("supported"):
        raise RuntimeError("GPT-OSS cannot start on the current setup: " + plan["reason"])
    # Pass resolved paths explicitly, so a later config change cannot redirect
    # this already-validated startup to another checkpoint.
    command += ["--model-dir", model, "--store-dir", store]
    return [sys.executable, "-m", "neural_runtime", *command], hardware, plan


def _port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _ready(port):
    request = Request(f"http://127.0.0.1:{port}/v1/models",
                      headers={"Authorization": "Bearer " + os.environ.get("NEURAL_API_KEY", "local")})
    try:
        with urlopen(request, timeout=1) as response:
            body = json.loads(response.read(64 * 1024))
        return any(item.get("id") == DEFAULT_API_MODEL and item.get("owned_by") == "neural-local"
                   for item in body.get("data", []) if isinstance(item, dict))
    except (OSError, URLError, ValueError, AttributeError, TypeError):
        return False


def _startup_announced(log_path, port):
    """Only our child can write this newly created startup log.

    The inherited server prints this banner after successfully binding its
    socket. A matching listener on its own is insufficient: another server
    could take the port while our child is still loading the model.
    """
    try:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 65536))
            tail = stream.read().decode("utf-8", errors="replace")
        return (f" base URL : http://127.0.0.1:{port}/v1" in tail
                and f" Neural OpenAI-compatible server  |  model id: {DEFAULT_API_MODEL}" in tail)
    except OSError:
        return False


def _stop_owned(process):
    """Never stop an existing server; only the child created by this launcher."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        # The Windows venv executable may spawn a base-Python child. Killing
        # only the venv parent would leak the model process and its GPU memory.
        import psutil
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
        if result.returncode and process.poll() is None:
            raise RuntimeError(f"Neural could not stop its server (process {process.pid}).")
        # taskkill may return before a descendant has released its listener
        # and CUDA allocations. Wait for the captured owned processes too.
        _, alive = psutil.wait_procs(descendants, timeout=15)
        if alive:
            raise RuntimeError("Neural's server is still shutting down (processes "
                               + ", ".join(str(child.pid) for child in alive) + ").")
    else:
        process.terminate()
    process.wait(timeout=15)


def run_session(command, port, *, timeout=900):
    if not _port_available(port):
        raise RuntimeError(f"Port {port} is already in use. If Neural is already running, open "
                           "start_neural_chat.bat; otherwise use start_neural.bat --port 8003.")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"neural-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
    print(f"Starting {DEFAULT_MODEL_LABEL}. The first start can take several minutes.", flush=True)
    print(f"Startup details: {log_path}", flush=True)
    # Share the launcher's existing console without opening another window.
    # A separate group isolates chat Ctrl+C, while closing the console sends
    # CTRL_CLOSE to the server too. CREATE_NO_WINDOW would orphan it on X.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=flags)
        try:
            started = last_notice = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Neural stopped during startup. Read {log_path} for the cause.")
                if _startup_announced(log_path, port) and _ready(port):
                    # A listener becoming ready does not override a failed child.
                    if process.poll() is not None:
                        raise RuntimeError(f"Neural could not take port {port}. Read {log_path}.")
                    break
                now = time.monotonic()
                if now - started > timeout:
                    raise RuntimeError(f"Startup timed out. Read {log_path} for progress and errors.")
                if now - last_notice >= 20:
                    print("Still preparing GPT-OSS; the chat will open when it is ready.", flush=True)
                    last_notice = now
                time.sleep(0.5)
            print(f"Ready at http://127.0.0.1:{port}/v1. Type /quit to close chat and stop this server.", flush=True)
            return subprocess.call([sys.executable, str(ROOT / "chat_client.py"),
                                    "--url", f"http://127.0.0.1:{port}/v1"], cwd=ROOT)
        finally:
            _stop_owned(process)


def start_neural(args):
    command, hardware, plan = prepare_launch(args)
    print(f"Default model: {DEFAULT_MODEL_LABEL}")
    print(f"Graphics card: {hardware.get('gpu_name') or 'unavailable'}; context: {plan['context']:,} tokens.")
    if args.check:
        print("Setup check passed. No model was loaded. Double-click start_neural.bat to open chat.")
        return 0
    try:
        return run_session(command, args.port)
    except KeyboardInterrupt:
        print("\nNeural closed.")
        return 0
