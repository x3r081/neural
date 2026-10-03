"""Replay frozen prompts to one running engine; orchestration is deliberately serial."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def post(base, path, data):
    body = json.dumps(data, ensure_ascii=False).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('NEURAL_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['NEURAL_API_KEY']
    request = urllib.request.Request(base.rstrip('/')+path, data=body,
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--repeat', type=int, default=1)
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error('--repeat must be at least 1')
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be at least 1')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_requests = Path(args.requests).read_bytes()
    request_doc = json.loads(raw_requests)
    requests = request_doc.get('requests') if isinstance(request_doc, dict) else None
    if not isinstance(requests, list) or not requests:
        raise ValueError('requests file must contain a non-empty requests list')
    cohort_count = len(requests)
    if args.limit is not None:
        requests = requests[:args.limit]
    selected_count = len(requests)
    if not requests:
        raise ValueError('selected benchmark cohort is empty')
    record = {'status': 'running', 'engine': args.label, 'url': args.url,
              'requests_file_sha256': hashlib.sha256(raw_requests).hexdigest(),
              'scope': 'fixed raw prompts; cold conversation state per request; expert/page cache may be warm; no cross-engine output-quality equivalence assumed',
              'requested_cohort_count': cohort_count,
              'selected_cohort_count': selected_count,
              'repeat_count': args.repeat,
              'runs': []}
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    try:
        for entry in requests * args.repeat:
            prompt = entry['prompt']
            expected = entry['prompt_ids']
            tokenized = post(args.url, '/tokenize', {'content': prompt, 'add_special': False})['tokens']
            if tokenized != expected:
                raise AssertionError(f"Prompt token mismatch for {entry['id']}: {len(tokenized)} vs {len(expected)}")
            payload = {'prompt': prompt, 'n_predict': entry['max_tokens'], 'temperature': 0,
                       'seed': 0, 'cache_prompt': False, 'stream': False,
                       'return_tokens': True,
                       'top_k': 20, 'top_p': 0.95}
            before = time.perf_counter()
            response = post(args.url, '/completion', payload)
            wall = time.perf_counter()-before
            observed_prompt_n = response.get('timings', {}).get('prompt_n')
            if observed_prompt_n != len(expected):
                raise AssertionError(f"Actual prefill count mismatch: {observed_prompt_n} vs {len(expected)}")
            run = {'id': entry['id'], 'request': payload, 'prompt_token_ids_match': True,
                   'client_wall_s': wall, 'response': response}
            record['runs'].append(run)
            output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps({'id': entry['id'], 'client_wall_s': wall,
                              'timings': response.get('timings')}), flush=True)
        if selected_count < cohort_count:
            record['status'] = 'partial'
        elif args.repeat > 1:
            # Repeated IDs are intentionally not accepted by the single-pass
            # comparison summarizer; keep this distinct from a canonical run.
            record['status'] = 'repeated'
        else:
            record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        output.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
