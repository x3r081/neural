$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'F:\AI\Neural\experiments\neuralserver-opt'
$result = 'benchmarks\codex_decode_followup_20261001\persistent_cpu_frozen.json'
if (Test-Path -LiteralPath $result) { throw "Result already exists: $result" }
$env:NEURAL_PREFILL_GROUP = '0'
$env:NEURAL_PREFILL_GROUP_EXPERTS = '3'
$env:NEURAL_PREFILL_GROUP_ROWS = '1024'
& 'F:\AI\Neural\.venv\Scripts\python.exe' tools\bench_vs_llama.py $result `
    --schedule off,cpu,cpu,off `
    --kinds 'off=--kdll gptoss_cpu_cap2.dll --kernel-persistent 0' 'cpu=--kdll gptoss_cpu_persistent.dll --kernel-persistent 1' `
    --replay-prompts 'benchmarks\codex_decode_followup_20261001\persistent_cpu_mirrored.json.replay.json' `
    --workload conversation --ctx-tokens 13000 --max-tokens 512 --effort low `
    --ctx-root 'F:\AI\Neural\experiments\neuralserver-base' `
    --python 'F:\AI\Neural\.venv\Scripts\python.exe' `
    --log-dir 'logs\codex_persistent_frozen' `
    --neural-args '--pool 6.05 --smax 16384 --threads 8 --hotset hotset_code.json --refresh-every 0 --refresh-m 16 --capbufs 16 --prefill-m 64 --early-every 4 --early-tokens 32 --static 250 --skip-quality --kv-ring 256 --scratch 6 --kernel-fuse 1 --prefill-order layer --store-dir G:\NeuralStores\gptoss120b_ps4 --grouped-prefill-gemm 0 --trim-prefill-cache 0 --masked-gemv 0'
exit $LASTEXITCODE
