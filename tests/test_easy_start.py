"""The beginner launcher never selects an experimental model or owns other servers."""
import json
import socket
import sys
from types import SimpleNamespace

import pytest

from neural_runtime import __main__ as cli, launcher
from neural_runtime.defaults import default_paths


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("NEURAL_MODEL_DIR", raising=False)
    monkeypatch.delenv("NEURAL_STORE_DIR", raising=False)
    return tmp_path


def test_fresh_user_gets_a_gptoss_install_instruction(isolated_config):
    with pytest.raises(ValueError, match="GPT-OSS 120B is the default.*install_neural.bat"):
        cli.model_paths(cli.parser().parse_args(["serve"]))


def test_default_model_paths_need_no_handwritten_config(isolated_config):
    paths = default_paths(isolated_config / "data")
    paths["model_dir"].mkdir(parents=True)
    (paths["model_dir"] / "config.json").write_text("{}")
    assert cli.model_paths(cli.parser().parse_args(["serve"])) == (
        str(paths["model_dir"]), str(paths["store_dir"]))


def test_existing_paths_and_advanced_explicit_model_are_preserved(isolated_config):
    model, store = isolated_config / "existing", isolated_config / "store"
    (isolated_config / "neural.local.json").write_text(json.dumps({"model_dir": str(model), "store_dir": str(store)}))
    assert cli.model_paths(cli.parser().parse_args(["serve"])) == (str(model), str(store))
    explicit = isolated_config / "explicit"
    assert cli.model_paths(cli.parser().parse_args(["serve", "--model-dir", str(explicit)]))[0] == str(explicit)


def test_local_relative_paths_resolve_from_repository_not_callers_directory(isolated_config, monkeypatch, tmp_path):
    (isolated_config / "neural.local.json").write_text('{"model_dir":"data/model","store_dir":"data/store"}')
    monkeypatch.chdir(isolated_config.parent)
    model, store = cli.model_paths(cli.parser().parse_args(["serve"]))
    assert model == str(isolated_config / "data" / "model")
    assert store == str(isolated_config / "data" / "store")


def test_beginner_start_forces_fast_gptoss_and_resolved_paths(monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "model_paths", lambda args: ("M", "S"))
    def inspect(model, store, *, mode):
        observed.append((model, store, mode))
        return SimpleNamespace(to_dict=lambda: {"backend": "gptoss-native-mxfp4"})
    monkeypatch.setattr("neural_runtime.model_spec.inspect_model", inspect)
    monkeypatch.setattr(cli, "native_plan", lambda *a: ({"gpu_name": "Test"}, {"supported": True, "context": 16384}))
    command, _, _ = launcher.prepare_launch(cli.parser().parse_args(["start"]))
    assert observed == [("M", "S", "fast")]
    assert command[command.index("--mode") + 1] == "fast"
    assert command[-4:] == ["--model-dir", "M", "--store-dir", "S"]


def test_beginner_start_rejects_reference_adapter(monkeypatch):
    monkeypatch.setattr(cli, "model_paths", lambda args: ("M", None))
    monkeypatch.setattr("neural_runtime.model_spec.inspect_model", lambda *a, **kw:
                        SimpleNamespace(to_dict=lambda: {"backend": "hf-reference"}))
    monkeypatch.setattr(cli, "native_plan", lambda *a: pytest.fail("must reject before planning"))
    with pytest.raises(ValueError, match="requires GPT-OSS"):
        launcher.prepare_launch(cli.parser().parse_args(["start"]))


def test_start_check_never_launches_server_or_chat(monkeypatch):
    monkeypatch.setattr(launcher, "prepare_launch", lambda args: (
        [], {"gpu_name": "Test"}, {"supported": True, "context": 16384}))
    monkeypatch.setattr(launcher, "run_session", lambda *a: pytest.fail("check must not load a model"))
    assert cli.main(["start", "--check"]) == 0


def test_occupied_port_never_starts_or_stops_existing_process(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_port_available", lambda p: False)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("port in use"))
    monkeypatch.setattr(launcher, "_stop_owned", lambda p: pytest.fail("no process owned"))
    with pytest.raises(RuntimeError, match="already in use"):
        launcher.run_session(["python"], 8001)
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("chat_failure", [False, True])
def test_chat_exit_and_failure_both_stop_only_owned_server(monkeypatch, tmp_path, chat_failure):
    process = SimpleNamespace(poll=lambda: None, pid=12345)
    stopped = []
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_port_available", lambda p: True)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(launcher, "_ready", lambda p: True)
    monkeypatch.setattr(launcher, "_startup_announced", lambda *a: True)
    monkeypatch.setattr(launcher, "_stop_owned", lambda p: stopped.append(p))
    def chat(*args, **kwargs):
        if chat_failure:
            raise OSError("chat could not open")
        return 0
    monkeypatch.setattr(launcher.subprocess, "call", chat)
    if chat_failure:
        with pytest.raises(OSError, match="chat could not open"):
            launcher.run_session(["python"], 8001)
    else:
        assert launcher.run_session(["python"], 8001) == 0
    assert stopped == [process]


def test_startup_timeout_cleans_up_without_opening_chat(monkeypatch, tmp_path):
    process = SimpleNamespace(poll=lambda: None, pid=12345)
    stopped = []
    times = iter([0, 2])
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_port_available", lambda p: True)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(launcher, "_ready", lambda p: False)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(launcher.subprocess, "call", lambda *a, **kw: pytest.fail("not ready"))
    monkeypatch.setattr(launcher, "_stop_owned", lambda p: stopped.append(p))
    with pytest.raises(RuntimeError, match="timed out"):
        launcher.run_session(["python"], 8001, timeout=1)
    assert stopped == [process]


def test_foreign_matching_listener_cannot_satisfy_child_readiness(monkeypatch, tmp_path):
    process = SimpleNamespace(poll=lambda: None, pid=12345)
    stopped = []
    times = iter([0, 2])
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_port_available", lambda p: True)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(launcher, "_ready", lambda p: True)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(launcher.subprocess, "call", lambda *a, **kw: pytest.fail("foreign listener"))
    monkeypatch.setattr(launcher, "_stop_owned", lambda p: stopped.append(p))
    with pytest.raises(RuntimeError, match="timed out"):
        launcher.run_session(["python"], 8001, timeout=1)
    assert stopped == [process]


def test_child_banner_matches_only_its_bound_model_and_port(tmp_path):
    log = tmp_path / "startup.log"
    log.write_text(" Neural OpenAI-compatible server  |  model id: gpt-oss-120b-neural\n"
                   " base URL : http://127.0.0.1:8001/v1\n")
    assert launcher._startup_announced(log, 8001)
    assert not launcher._startup_announced(log, 8003)


def test_real_background_server_is_reaped_after_chat_exit(tmp_path, monkeypatch):
    """Exercise Windows venv parent/child cleanup with a tiny HTTP stub, no model."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    stub = tmp_path / "stub_server.py"
    stub.write_text('''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data":[{"id":"gpt-oss-120b-neural","owned_by":"neural-local"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
port = int(sys.argv[1])
server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
print(" Neural OpenAI-compatible server  |  model id: gpt-oss-120b-neural")
print(f" base URL : http://127.0.0.1:{port}/v1", flush=True)
server.serve_forever()
''')
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher.subprocess, "call", lambda *a, **kw: 0)
    assert launcher.run_session([sys.executable, str(stub), str(port)], port, timeout=15) == 0
    with socket.socket() as check:
        check.settimeout(1)
        assert check.connect_ex(("127.0.0.1", port)) != 0
