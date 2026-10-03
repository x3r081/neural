"""Verify original downloaded bytes against the pinned Hugging Face LFS digests."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=r'F:\Models\Qwen3.6-35B-A3B')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root, output = Path(args.model_dir), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((root/'neural_download_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'downloaded_size_verified'
    assert receipt['repo_id'] == 'Qwen/Qwen3.6-35B-A3B'
    assert receipt['revision'] == '995ad96eacd98c81ed38be0c5b274b04031597b0'
    result = {'repo_id': receipt['repo_id'], 'revision': receipt['revision'],
              'status': 'running', 'files': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for entry in receipt['files']:
        path = root/entry['name']
        assert path.stat().st_size == entry['bytes'], str(path)
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if entry['sha256']:
            assert digest == entry['sha256'], f'Hash mismatch: {path}'
        result['files'].append(dict(entry, actual_sha256=digest,
                                    lfs_digest_verified=bool(entry['sha256'])))
        print(entry['name'] + ' SHA256_OK' if entry['sha256'] else entry['name'] + ' recorded', flush=True)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    result.update(status='verified', elapsed_s=time.perf_counter()-started)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
