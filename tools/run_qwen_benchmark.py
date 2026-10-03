"""Run one owned server and a frozen HTTP suite, then always stop that server."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from benchmark_qwen_http import post


def _same_process(process, create_time):
    """Check a saved psutil handle still identifies the same PID incarnation."""
    try:
        return process.is_running() and process.create_time() == create_time
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _owned_process_tree(root_pid, root_create_time):
    """Return identity-checked references for the launcher and its descendants."""
    try:
        root = psutil.Process(root_pid)
        if not _same_process(root, root_create_time):
            return []
        descendants = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []
    refs = [(root, root_create_time)]
    seen = {(root.pid, root_create_time)}
    for child in descendants:
        try:
            created = child.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        identity = (child.pid, created)
        if identity not in seen:
            refs.append((child, created))
            seen.add(identity)
    return refs


def _sample_owned_rss(refs):
    total = 0
    largest = None
    for process, created in refs:
        if not _same_process(process, created):
            continue
        try:
            rss = process.memory_info().rss
            name = process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        total += rss
        if largest is None or rss > largest['rss_bytes']:
            largest = {'pid': process.pid, 'rss_bytes': rss, 'name': name}
    return total, largest


def _stop_owned_process_tree(root_pid, root_create_time, observed_refs, popen):
    """Stop only process incarnations observed beneath this Popen-owned root."""
    refs = list(observed_refs.values())
    for process, created in _owned_process_tree(root_pid, root_create_time):
        observed_refs[(process.pid, created)] = (process, created)
    refs = list(observed_refs.values())
    # Children first so a launcher cannot immediately respawn its inference child.
    targets = [entry for entry in reversed(refs)
               if entry[0].pid != root_pid]
    root_refs = [entry for entry in refs if entry[0].pid == root_pid]
    targets.extend(root_refs)
    live = []
    for process, created in targets:
        if _same_process(process, created):
            live.append((process, created))
            try:
                process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    if live:
        try:
            psutil.wait_procs([process for process, _ in live], timeout=15)
        except psutil.Error:
            pass
    remaining = [(process, created) for process, created in live
                 if _same_process(process, created)]
    for process, created in remaining:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if remaining:
        try:
            psutil.wait_procs([process for process, _ in remaining], timeout=15)
        except psutil.Error:
            pass
    try:
        popen.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return [(process.pid, created) for process, created in refs
            if _same_process(process, created)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['llama', 'neural'], required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cpu-moe-layers', type=int, default=38)
    ap.add_argument('--gdn-backend', choices=['torch', 'fla'], default='torch')
    ap.add_argument('--expert-backend', choices=['staged', 'native'], default='native')
    ap.add_argument('--gpu-cache', type=float, default=0)
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--native-gpu-layers', type=int, default=0)
    ap.add_argument('--validate-http', action='store_true')
    ap.add_argument('--validation-only', action='store_true')
    ap.add_argument('--server-root', type=Path, default=ROOT,
                    help='Neural source directory; permits an immutable reference snapshot for A/B runs.')
    ap.add_argument('--recurrent-graph', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--native-workspace', action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    server_root = args.server_root.resolve()
    if args.engine != 'neural' and server_root != ROOT:
        ap.error('--server-root is only supported for Neural')
    if args.engine == 'neural' and not (server_root/'qwen_server.py').is_file():
        ap.error('--server-root must contain qwen_server.py')
    if args.repeat < 1:
        ap.error('--repeat must be at least 1')
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be at least 1')
    if args.validation_only and not args.validate_http:
        ap.error('--validation-only requires --validate-http')
    request_doc = json.loads(Path(args.requests).read_text(encoding='utf-8'))
    request_entries = request_doc.get('requests') if isinstance(request_doc, dict) else None
    if not isinstance(request_entries, list) or not request_entries:
        ap.error('--requests must contain a non-empty requests list')
    request_count = len(request_entries)
    selected_count = min(args.limit, request_count) if args.limit is not None else request_count
    dest = ROOT / 'benchmarks' / 'qwen36_20261002' / args.name
    dest.mkdir(exist_ok=False)
    port = 8002 if args.engine == 'llama' else 8001
    url = f'http://127.0.0.1:{port}'
    if args.engine == 'llama':
        cmd = [str(ROOT / '.tools/llama-b11349-win-cuda-12.4/llama-server.exe'),
               '-m', r'F:\Models\Qwen3.6-35B-A3B-bf16.gguf',
               '--host', '127.0.0.1', '--port', str(port), '-ngl', '99',
               '-ncmoe', str(args.cpu_moe_layers), '-t', str(args.threads),
               '-tb', str(args.threads), '-c', str(args.context), '-np', '1',
               '-b', '512', '-ub', '128', '-fa', 'on', '-ctk', 'bf16', '-ctv', 'bf16',
               '--jinja']
    else:
        cmd = [sys.executable, '-u', str(server_root / 'qwen_server.py'), '--port', str(port),
               '--threads', str(args.threads), '--context', str(args.context),
               '--expert-backend', args.expert_backend, '--gpu-expert-budget-gib', str(args.gpu_cache),
               '--native-gpu-layers', str(args.native_gpu_layers),
               '--gdn-backend', args.gdn_backend]
        for name in ('recurrent_graph', 'native_workspace'):
            value = getattr(args, name)
            if value is not None:
                cmd.append('--' + ('' if value else 'no-') + name.replace('_', '-'))
    record = {'status': 'starting', 'engine': args.engine, 'command': cmd,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model_revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
              'requested_cohort_count': request_count,
              'selected_cohort_count': selected_count,
              'repeat_count': args.repeat,
              'resources': {'min_available_ram_bytes': psutil.virtual_memory().available,
                            'max_server_rss_bytes': 0,
                            'max_owned_process_tree_rss_bytes': 0,
                            'max_owned_process_rss_bytes': 0,
                            'max_owned_process_pid': None,
                            'max_owned_process_name': None,
                            'max_server_rss_bytes_scope': (
                                'sum of RSS for the launcher and observed live descendants; '
                                'shared pages may be counted in each process RSS'
                            ),
                            'owned_process_rss_scope': (
                                'sum of RSS for the launcher and observed live descendants; '
                                'shared pages may be counted in each process RSS'
                            ),
                            'owned_process_tree_samples': 0}}
    record['weight_precision'] = 'bf16'
    record['kv_cache_precision'] = 'bf16'
    record['gdn_recurrent_state_precision'] = 'float32'
    record['precision_evidence'] = {
        'weights': 'checkpoint_verification.json and gguf_verification.json',
        'neural_caches': 'numerics01.json',
        'llama_caches': 'explicit -ctk bf16 -ctv bf16 and upstream F32 recurrent state',
        'note': 'GGUF preserves original BF16 values; its 301 F32 tensors use the verified converter transforms',
    }
    record['runtime_thread_environment'] = {name: os.environ.get(name) for name in (
        'GOMP_SPINCOUNT', 'OMP_WAIT_POLICY', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'KMP_BLOCKTIME')}
    record['source_git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    record['server_source_root'] = str(server_root)
    if (server_root/'snapshot_manifest.json').is_file():
        record['server_source_snapshot'] = json.loads((server_root/'snapshot_manifest.json').read_text(encoding='utf-8'))
    record['runtime_file_sha256'] = {
        name: hashlib.sha256((server_root/name).read_bytes()).hexdigest()
        for name in ('qwen_server.py', 'qwen_neural/backend.py', 'qwen_neural/generation.py',
                     'qwen_neural/native_dispatch.py', 'qwen_neural/cpu_experts.py',
                     'qwen_neural/recurrent_graph.py', 'kernels/qwen_bf16_experts.c',
                     'qwen_bf16_experts.dll')
        if (server_root/name).is_file()
    } if args.engine == 'neural' else {}
    manifest = dest / 'run.json'
    def save():
        manifest.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    save()
    # Refuse to collide with an existing service; never terminate an unrelated PID.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} is already occupied')
    proc = None
    server_root_create_time = None
    owned_refs = {}
    owned_refs_lock = threading.Lock()
    monitor_thread = None
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.wait(1):
            try:
                resource = record['resources']
                resource['min_available_ram_bytes'] = min(resource['min_available_ram_bytes'], psutil.virtual_memory().available)
                refs = _owned_process_tree(proc.pid, server_root_create_time)
                with owned_refs_lock:
                    for process, created in refs:
                        owned_refs[(process.pid, created)] = (process, created)
                total_rss, largest = _sample_owned_rss(refs)
                resource['owned_process_tree_samples'] += 1
                resource['max_server_rss_bytes'] = max(resource['max_server_rss_bytes'], total_rss)
                resource['max_owned_process_tree_rss_bytes'] = max(
                    resource['max_owned_process_tree_rss_bytes'], total_rss)
                if largest and largest['rss_bytes'] > resource['max_owned_process_rss_bytes']:
                    resource['max_owned_process_rss_bytes'] = largest['rss_bytes']
                    resource['max_owned_process_pid'] = largest['pid']
                    resource['max_owned_process_name'] = largest['name']
            except psutil.Error:
                continue
    try:
        with (dest / 'server.log').open('w', encoding='utf-8') as log:
            before = time.perf_counter()
            proc = subprocess.Popen(cmd, cwd=server_root, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            record['server_pid'] = proc.pid
            root_process = psutil.Process(proc.pid)
            server_root_create_time = root_process.create_time()
            with owned_refs_lock:
                owned_refs[(root_process.pid, server_root_create_time)] = (
                    root_process, server_root_create_time)
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            while time.perf_counter()-before < 600:
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited {proc.returncode}; see server.log')
                try:
                    health_headers = {}
                    if os.environ.get('NEURAL_API_KEY'):
                        health_headers['Authorization'] = 'Bearer ' + os.environ['NEURAL_API_KEY']
                    health_request = urllib.request.Request(url+'/health', headers=health_headers)
                    with urllib.request.urlopen(health_request, timeout=2) as response:
                        if response.status == 200:
                            record['health'] = json.load(response)
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError('Server did not become ready in 600 seconds')
            record['startup_s'] = time.perf_counter()-before
            record['gpu_memory_at_ready_csv'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader'], text=True).strip()
            record['status'] = 'warmup'
            save()
            entry = request_entries[0]
            record['warmup'] = post(url, '/completion', {'prompt': entry['prompt'], 'n_predict': 8,
                                    'temperature': 0, 'cache_prompt': False, 'seed': 0})
            record['status'] = 'benchmarking'
            save()
            bench_cmd = [sys.executable, '-u', str(ROOT/'tools/benchmark_qwen_http.py'),
                         '--url', url, '--label', args.name, '--requests', str(Path(args.requests).resolve()),
                         '--output', str(dest/'results.json'), '--repeat', str(args.repeat)]
            if args.limit is not None:
                bench_cmd += ['--limit', str(args.limit)]
            if not args.validation_only:
                subprocess.run(bench_cmd, cwd=ROOT, check=True)
            record['gpu_memory_after_benchmark_csv'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader'], text=True).strip()
            with urllib.request.urlopen(urllib.request.Request(url+'/health', headers=health_headers), timeout=30) as response:
                record['health_after_benchmark'] = json.load(response)
            if args.validate_http:
                if args.engine != 'neural':
                    raise ValueError('--validate-http currently targets the Neural API')
                subprocess.run([sys.executable, '-u', str(ROOT/'tools/validate_qwen_http.py'),
                    '--url', url, '--output', str(dest/'http_validation.json')], cwd=ROOT, check=True)
            if args.validation_only:
                record['status'] = 'validation_only'
            elif selected_count < request_count:
                record['status'] = 'partial'
            elif args.repeat > 1:
                record['status'] = 'repeated'
            else:
                record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join()
        remaining_owned = []
        if proc is not None and server_root_create_time is not None:
            with owned_refs_lock:
                observed = dict(owned_refs)
            remaining_owned = _stop_owned_process_tree(
                proc.pid, server_root_create_time, observed, proc)
        record['remaining_owned_processes'] = [
            {'pid': pid, 'create_time': created} for pid, created in remaining_owned]
        record['server_stopped'] = proc is None or (
            proc.poll() is not None and not remaining_owned)
        record['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
