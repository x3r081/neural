"""Capture and compare original vs ported Neural logits and cache state.

Run captures serially in separate processes, with the same project venv and
checkpoint. This is correctness instrumentation, not a performance benchmark.
Tensor payloads belong in .tools; only compact JSON evidence is checked in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def cache_hashes(cache):
    import torch
    result = {}

    def visit(name, value):
        if isinstance(value, torch.Tensor):
            cpu = value.detach().to('cpu').contiguous()
            result[name] = {
                'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'finite': bool(torch.isfinite(cpu).all()),
            }
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(f'{name}.{i}', child)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f'{name}.{key}', child)

    for i, layer in enumerate(cache.layers):
        for name, value in vars(layer).items():
            visit(f'layers.{i}.{name}', value)
    if not result or not all(entry['finite'] for entry in result.values()):
        raise ValueError('Cache inspection found no tensors or non-finite state')
    return result


def capture(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    from qwen_neural.backend import QwenBackend

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    frozen = json.loads(args.requests.read_text(encoding='utf-8'))['requests']
    token_source = json.loads(args.continuations.read_text(encoding='utf-8'))['runs']
    continuations = {run['id']: run['response']['tokens'][:args.tokens] for run in token_source}
    kwargs = dict(expert_backend='native', gdn_backend='torch', native_threads=8,
                  native_dispatch='grouped', native_gpu_layers=2, gpu_expert_budget_gib=3)
    if args.ported:
        kwargs.update(native_workspace=True, recurrent_graph=args.recurrent_graph)
    report = {
        'source_root': str(args.source_root.resolve()), 'ported': args.ported,
        'recurrent_graph': args.recurrent_graph, 'tokens_per_case': args.tokens,
        'checkpoint_revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
        'runtime_file_sha256': {
            str(path.relative_to(args.source_root.resolve())): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.source_root.resolve()/'qwen_neural').glob('*.py'))
        },
        'requests_sha256': hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        'continuations_sha256': hashlib.sha256(args.continuations.read_bytes()).hexdigest(),
        'scope': 'Teacher-forced full-vocabulary logits and exact cache-state hashes; not throughput or a general quality score.',
        'cases': {},
    }
    payload = {}
    from qwen_neural.cpu_experts import _dll_path
    report['native_dll_path'] = str(_dll_path())
    report['native_dll_sha256'] = hashlib.sha256(_dll_path().read_bytes()).hexdigest()
    with QwenBackend(r'F:\Models\Qwen3.6-35B-A3B', **kwargs) as backend:
        with torch.inference_mode():
            for entry in frozen:
                ids = entry['prompt_ids']
                continuation = continuations[entry['id']]
                if len(continuation) != args.tokens:
                    raise ValueError('Frozen continuation is shorter than requested')
                past = None
                for start in range(0, len(ids), 64):
                    out = backend.forward(torch.tensor([ids[start:start+64]], device=backend.device),
                                          past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                logits = [out.logits[:, -1, :].detach().cpu()]
                states = {'prefill': cache_hashes(past)}
                for token in continuation:
                    out = backend.forward(torch.tensor([[token]], device=backend.device),
                                          past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
                    logits.append(out.logits[:, -1, :].detach().cpu())
                states['final'] = cache_hashes(past)
                payload[entry['id']] = torch.cat(logits)
                report['cases'][entry['id']] = {
                    'prompt_ids': ids, 'continuation_ids': continuation,
                    'logit_positions': len(logits), 'cache_state': states,
                }
                print('Captured', entry['id'], len(logits), 'positions', flush=True)
                del past, out, logits
        report['execution'] = backend.memory_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.tensors)
    report['tensor_sha256'] = hashlib.sha256(args.tensors.read_bytes()).hexdigest()
    report['status'] = 'CAPTURED_FOR_COMPARISON'
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def compare(args):
    import torch
    ref = json.loads(args.reference.read_text(encoding='utf-8'))
    cand = json.loads(args.candidate.read_text(encoding='utf-8'))
    if any(report.get('status') != 'CAPTURED_FOR_COMPARISON' for report in (ref, cand)):
        raise ValueError('Both captures must have completed successfully')
    if ref.get('ported') is not False or cand.get('ported') is not True:
        raise ValueError('Expected original reference and explicitly ported candidate')
    if ref.get('native_dll_sha256') != '11076a97409428a22765425e7bdeda05091114dec1256665a9d4b4b4e897b3bf':
        raise ValueError('Reference is not the preserved original CPU kernel')
    for field in ('checkpoint_revision', 'requests_sha256', 'continuations_sha256', 'tokens_per_case'):
        if ref[field] != cand[field]:
            raise ValueError(f'Comparison metadata mismatch: {field}')
    for report, path in ((ref, args.reference_tensors), (cand, args.candidate_tensors)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['tensor_sha256']:
            raise ValueError(f'Tensor integrity mismatch: {path}')
    r = torch.load(args.reference_tensors, map_location='cpu', weights_only=True)
    c = torch.load(args.candidate_tensors, map_location='cpu', weights_only=True)
    if set(r) != set(c) or set(ref['cases']) != set(cand['cases']):
        raise ValueError('Different case sets')
    if not r or set(r) != set(ref['cases']):
        raise ValueError('Empty or inconsistent tensor/report cases')
    cases = {}
    for name in r:
        for field in ('prompt_ids', 'continuation_ids'):
            if ref['cases'][name][field] != cand['cases'][name][field]:
                raise ValueError(f'Different {field} in {name}')
        if r[name].shape != c[name].shape or r[name].dtype != c[name].dtype:
            raise ValueError(f'Logit shape/dtype differs in {name}')
        if r[name].ndim != 2 or r[name].shape[0] != ref['tokens_per_case'] + 1:
            raise ValueError(f'Unexpected logit position count in {name}')
        cases[name] = {
            'logit_positions': r[name].shape[0],
            'identical_logits': bool(torch.equal(r[name], c[name])),
            'all_finite': bool(torch.isfinite(r[name]).all() and torch.isfinite(c[name]).all()),
            'max_abs_logit_difference': float((r[name].float()-c[name].float()).abs().max()),
            'argmax_matches': int((r[name].argmax(-1)==c[name].argmax(-1)).sum()),
            'identical_cache_states': ref['cases'][name]['cache_state'] == cand['cases'][name]['cache_state'],
        }
    passed = all(v['identical_logits'] and v['identical_cache_states'] and v['all_finite'] for v in cases.values())
    report = {'status': 'PASS_EXACT_LOGITS_AND_STATES' if passed else 'MISMATCH',
              'evidence_label': 'MEASURED', 'reference': str(args.reference),
              'candidate': str(args.candidate), 'cases': cases,
              'scope': 'Bounded teacher-forced numerical identity, not general answer quality.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--source-root', type=Path, default=ROOT)
    cap.add_argument('--ported', action='store_true')
    cap.add_argument('--recurrent-graph', action='store_true')
    cap.add_argument('--requests', type=Path, default=ROOT/'benchmarks/qwen36_20261002/requests_v1.json')
    cap.add_argument('--continuations', type=Path, default=ROOT/'benchmarks/qwen36_20261002/neural_torch_final02/results.json')
    cap.add_argument('--tokens', type=int, default=24)
    cap.add_argument('--output', type=Path, required=True)
    cap.add_argument('--tensors', type=Path, required=True)
    comp = sub.add_parser('compare')
    for option in ('reference', 'candidate', 'reference-tensors', 'candidate-tensors', 'output'):
        comp.add_argument('--'+option, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; use a fresh evidence path')
    if args.mode == 'capture':
        if args.tensors.exists():
            p.error('Tensor output already exists; use a fresh evidence path')
        if args.recurrent_graph and not args.ported:
            p.error('--recurrent-graph requires --ported')
        if args.tokens < 1:
            p.error('--tokens must be positive')
        capture(args)
        return 0
    return compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
