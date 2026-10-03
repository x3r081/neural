"""Offline replay of the server's VRAM admission rule (copy-on-compute, rate-limited) on a
recorded two-turn trace: prompt routing counts + per-token decode routing of each turn.

    python tools/admission_replay.py benchmarks/admission_trace_13k.json

The trace (13,150-token code prompt, 512-token answer, then a follow-up) was recorded with
--count-routing and POST/GET /neural/trace. The replay reproduces the server's measured VRAM
hit (turn 0: 0.157 measured / 0.162 replayed; turn 1: 0.329 / 0.331) and compares how the
prompt's counts enter the admission counts (SIMULATED):
  as-is      the prompt at full weight (the old behaviour)
  equiv:K    the prompt worth at most K generated tokens (--prefill-weight K, rounded down)
  seed:K     bulk-load the prompt's top-K experts into the adaptive slots first
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "benchmarks", "admission_trace_13k.json")))
cnt = np.array(json.load(open(os.path.join(ROOT, "hotset_code.json")))["counts"], dtype=np.int64).ravel()
order = np.argsort(-cnt, kind="stable")
S, U, NE = 250, 430, 128                     # static core, usable slots, experts per layer (server defaults)


def replay(ids, gen0, owner0, capbuf=16, lat=1, seed=0):
    owner = list(owner0)
    GEN = np.array(gen0, dtype=np.int64).ravel().copy()
    if seed:
        res = set(owner)
        cands = [int(g) for g in np.argsort(-GEN, kind="stable") if int(g) not in res and GEN[g] > 0]
        vict = sorted(range(S, U), key=lambda s: (GEN[owner[s]], cnt[owner[s]]))
        for g, s in zip(cands[:seed], vict):
            if GEN[g] > GEN[owner[s]]:
                owner[s] = g
    slot = {g: s for s, g in enumerate(owner)}
    CAND, pend, hits = set(), [], []

    def refresh(m):
        mk = 0
        for g in np.argsort(-GEN, kind="stable"):
            if GEN[g] < 2 or mk >= m:
                break
            g = int(g)
            if g not in slot and g not in CAND and all(g != p[1] for p in pend):
                CAND.add(g)
                mk += 1
    refresh(64)                                               # --prefill-m 64
    for t in range(len(ids)):
        still = []
        for rt, g, s in pend:                                 # captured experts land one token later
            if rt <= t:
                slot[g] = s
            else:
                still.append((rt, g, s))
        pend = still
        h = 0
        for L in range(36):
            cur = [L * NE + int(e) for e in ids[t][L]]
            for g in cur:
                GEN[g] += 1
                if g in slot:
                    h += 1
                    continue
                if g in CAND and len(pend) < capbuf:          # copy-on-compute admission
                    keys = [(GEN[owner[s]], cnt[owner[s]], s) for s in range(S, U)
                            if owner[s] not in cur and all(s != p[2] for p in pend)]
                    _, _, s = min(keys)
                    slot.pop(owner[s], None)
                    owner[s] = g
                    CAND.discard(g)
                    pend.append((t + lat, g, s))
        hits.append(h / 144)
        if (t + 1) % (4 if t + 1 <= 32 else 8) == 0:          # --early-every 4 / --refresh-every 8
            refresh(32)
    return float(np.mean(hits)), owner


def weighted(pc, n, mode):
    pc = np.array(pc, dtype=np.float64).ravel()
    if mode == "as-is":
        return pc.astype(np.int64)
    k = float(mode.split(":")[1])
    return np.floor(pc * min(1.0, k / max(n, 1))).astype(np.int64)


owner0 = [int(g) for g in order[:U]]
t0, t1 = d["turns"]
n0 = t0["usage"]["prompt_tokens"] - t0["usage"]["prompt_tokens_details"]["cached_tokens"]
n1 = t1["usage"]["prompt_tokens"] - t1["usage"]["prompt_tokens_details"]["cached_tokens"]
print(f"measured VRAM hit: turn 0 {t0['neural']['vram_hit']:.3f}, turn 1 {t1['neural']['vram_hit']:.3f}")
for mode in ("as-is", "equiv:0", "equiv:16", "equiv:64", "equiv:256"):
    h0, own = replay(t0["decode_ids"], weighted(t0["prefill_counts"], n0, mode), owner0)
    h1, _ = replay(t1["decode_ids"], weighted(t1["prefill_counts"], n1, mode), own)
    print(f"{mode:10s} replayed hit: turn 0 {h0:.3f}, turn 1 {h1:.3f}")
for k in (60, 180):
    h0, _ = replay(t0["decode_ids"], t0["prefill_counts"], owner0, seed=k)
    print(f"seed:{k:<5d} replayed hit: turn 0 {h0:.3f}")
