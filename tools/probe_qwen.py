"""Serial full-model smoke and native-vs-staged numerical comparison."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from qwen_neural.backend import QwenBackend
from qwen_neural.generation import GenerationEngine, render_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=r'F:\Models\Qwen3.6-35B-A3B')
    ap.add_argument('--output', required=True)
    ap.add_argument('--tokens', type=int, default=16)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--backend', choices=['staged', 'native'], default='staged')
    ap.add_argument('--gpu-cache', type=float, default=1)
    ap.add_argument('--gdn-backend', choices=['torch', 'fla'], default='torch')
    args = ap.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    report = {'status': 'running', 'evidence': 'MEASURED', 'runs': [],
              'source_checkpoint': 'Qwen/Qwen3.6-35B-A3B',
              'revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
              'purpose': 'smoke/numerical check, not an admitted performance comparison'}
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    backend = None
    try:
        before = time.perf_counter()
        backend = QwenBackend(args.model_dir, expert_backend=args.backend,
                              gdn_backend=args.gdn_backend,
                              gpu_expert_budget_gib=args.gpu_cache)
        report['load_seconds'] = time.perf_counter()-before
        report['memory_after_load'] = backend.memory_report()
        print(json.dumps({'loaded': True, 'load_seconds': report['load_seconds'],
                          'memory': report['memory_after_load']}), flush=True)
        engine = GenerationEngine(backend, context=4096, prefill_chunk=64)
        prompt, _ = render_prompt(backend.tokenizer, {'messages': [
            {'role': 'user', 'content': 'What is 6 multiplied by 7? Answer with the number only.'}],
            'chat_template_kwargs': {'enable_thinking': False}})
        modes = ['staged', 'native'] if args.compare else [args.backend]
        for mode in modes:
            backend.expert_backend = mode
            backend.reset()
            result = engine.generate(prompt, max_tokens=args.tokens, temperature=0)
            result['backend'] = mode
            result['memory'] = backend.memory_report()
            result['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
            report['runs'].append(result)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps({'backend': mode, 'text': result['content'], 'timings': result['timings']}), flush=True)
        if args.compare:
            report['complete_greedy_token_ids_equal'] = report['runs'][0]['tokens'] == report['runs'][1]['tokens']
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if backend:
            backend.close()
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
