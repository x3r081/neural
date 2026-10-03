"""Measure prompt-ID validation only; no model or GPU is loaded."""
import argparse
import json
from pathlib import Path
import time

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(r'F:\Models\Qwen3.6-35B-A3B', local_files_only=True)
    requests = json.loads(Path('benchmarks/qwen36_20261002/requests_v1.json').read_text())
    ids = requests['requests'][0]['prompt_ids']
    def original(values):
        return bool(values) and not any(not isinstance(i, int) or i < 0 or i >= len(tokenizer) for i in values)
    vocab_size = len(tokenizer)
    def cached(values):
        return bool(values) and not any(not isinstance(i, int) or i < 0 or i >= vocab_size for i in values)
    cases = [ids, [], [-1], [vocab_size], [vocab_size-1], ['invalid']]
    same = all(original(case) == cached(case) for case in cases)
    times = {'original': [], 'cached': []}
    for _ in range(3):
        for name, check in [('original', original), ('cached', cached)]:
            start = time.perf_counter()
            assert check(ids)
            times[name].append(time.perf_counter()-start)
    result = {'evidence': 'MEASURED', 'scope': 'CPU-only input validation, not inference throughput',
              'prompt_token_count': len(ids), 'tokenizer_class': type(tokenizer).__name__,
              'vocab_size': vocab_size, 'boundary_decisions_equal': same,
              'seconds': times, 'note': 'vocabulary size is immutable for the lifetime of GenerationEngine'}
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
