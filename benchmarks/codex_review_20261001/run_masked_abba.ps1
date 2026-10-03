$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'F:\AI\Neural\experiments\neuralserver-opt'
$result = 'benchmarks\codex_review_20261001\masked_gemv_abba.json'
if (Test-Path -LiteralPath $result) { throw "Result already exists: $result" }
& 'F:\AI\Neural\.venv\Scripts\python.exe' tools\bench_vs_llama.py $result `
    --schedule off,on,on,off `
    --kinds 'off=--masked-gemv 0' 'on=--masked-gemv 1' `
    --replay-first-session --workload conversation --ctx-tokens 13000 --max-tokens 512 --effort low `
    --ctx-root 'F:\AI\Neural\experiments\neuralserver-base' `
    --python 'F:\AI\Neural\.venv\Scripts\python.exe' `
    --log-dir 'logs\codex_masked_abba' `
    --neural-args '--pool 6.05 --smax 16384 --threads 8 --hotset hotset_code.json --refresh-every 8 --refresh-m 16 --capbufs 16 --prefill-m 64 --early-every 4 --early-tokens 32 --static 250 --skip-quality --kv-ring 256 --scratch 6 --kernel-fuse 1 --prefill-order layer --store-dir G:\NeuralStores\gptoss120b_ps4'
exit $LASTEXITCODE
