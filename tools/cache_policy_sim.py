"""Offline VRAM expert-cache study on a recorded decode routing trace.

Record a trace against a running server (any key if NEURAL_API_KEY is unset):
    curl -X POST http://127.0.0.1:8000/neural/trace -d "{\"on\": true}"
    ... run some code-generation requests ...
    curl http://127.0.0.1:8000/neural/trace > trace.json
Then:
    python tools/cache_policy_sim.py trace.json [--hotset hotset_code.json]

Reports (all SIMULATED from the trace):
  1. hit rate of LRU vs Belady (perfect future knowledge) for several slot counts and
     static-core sizes, admitting on every miss (upper bound, ignores admission cost);
  2. cost-aware admission rules: CPU expert reads per token = misses + 2 x admissions
     (an admission is a RAM read to capture the expert plus its upload);
  3. multi-token verification (speculative decoding): unique experts read per verify step
     of T consecutive tokens, i.e. the RAM-bandwidth price of checking T tokens at once.
"""
import argparse, heapq, json
from collections import OrderedDict, defaultdict, deque
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--hotset", default="hotset_code.json", help="counts [36,128] used to seed the cache")
A = ap.parse_args()
d = json.load(open(A.trace))
d = d.get("trace", d)
ids = np.array(d["ids"], dtype=np.int64)                       # [T, 36, 4]
T, NL, K = ids.shape
static_gids = list(d.get("static_gids") or [])
toks = [list((np.arange(NL)[:, None] * 128 + ids[t]).reshape(-1)) for t in range(T)]
flat = [g for tk in toks for g in tk]
N = len(flat)
hs = json.load(open(A.hotset))
cnt = np.array(hs["counts"], dtype=np.int64).ravel() if "counts" in hs else np.zeros(NL * 128, np.int64)
order = [int(g) for g in np.argsort(-cnt, kind="stable")]
print(f"trace: {T} tokens, {N} expert uses, {len(set(flat))} distinct experts of {NL * 128}; "
      f"server had {d.get('usable_slots')} usable slots, {len(static_gids)} static")


def lru(cap, n_static):
    stat = set(order[:n_static])
    c = OrderedDict((g, 0) for g in order[n_static:cap])
    miss = 0
    for g in flat:
        if g in stat:
            continue
        if g in c:
            c.move_to_end(g)
            continue
        miss += 1
        if cap > n_static:
            if len(c) >= cap - n_static:
                c.popitem(last=False)
            c[g] = 0
    return 1 - miss / N


def belady(cap, n_static):
    stat = set(order[:n_static])
    nxt, last = [N] * N, {}
    for i in range(N - 1, -1, -1):
        nxt[i] = last.get(flat[i], N)
        last[flat[i]] = i
    first = last                                                # after the loop: first use of each expert
    c = {g: first.get(g, N) for g in order[n_static:cap]}
    h = [(-v, g) for g, v in c.items()]
    heapq.heapify(h)
    miss = 0
    for i, g in enumerate(flat):
        if g in stat:
            continue
        if g in c:
            c[g] = nxt[i]
            heapq.heappush(h, (-nxt[i], g))
            continue
        miss += 1
        if cap <= n_static:
            continue
        if len(c) < cap - n_static:
            c[g] = nxt[i]
            heapq.heappush(h, (-nxt[i], g))
            continue
        while True:                                             # lazy-invalidated max-heap of next use
            nv, v = h[0]
            if v in c and c[v] == -nv:
                break
            heapq.heappop(h)
        if -h[0][0] > nxt[i]:                                   # evict the farthest; bypass if we are farther
            heapq.heappop(h)
            del c[v]
            c[g] = nxt[i]
            heapq.heappush(h, (-nxt[i], g))
    return 1 - miss / N


def gated(cap, n_static, gate, window=64, k=2, r=2, horizon=64, oracle_evict=False):
    stat = set(order[:n_static])
    c = OrderedDict((g, 0) for g in order[n_static:cap])
    fut = defaultdict(deque)
    for t, tk in enumerate(toks):
        for g in tk:
            fut[g].append(t)
    hist, hcnt = deque(), defaultdict(int)
    miss = adm = 0
    for t, tk in enumerate(toks):
        for g in tk:
            fut[g].popleft()
            if g in stat or g in c:
                if g in c:
                    c.move_to_end(g)
                continue
            miss += 1
            if gate == "freq":
                ok = hcnt[g] + 1 >= k
            elif gate == "oracle":
                ok = sum(1 for tt in fut[g] if tt - t <= horizon) >= r
            else:
                ok = False
            if ok and cap > n_static:
                if len(c) >= cap - n_static:
                    if oracle_evict:
                        del c[max(c, key=lambda x: fut[x][0] if fut[x] else 1 << 30)]
                    else:
                        c.popitem(last=False)
                c[g] = 0
                adm += 1
        for g in tk:
            hist.append((t, g))
            hcnt[g] += 1
        while hist and hist[0][0] <= t - window:
            hcnt[hist.popleft()[1]] -= 1
    return 1 - miss / N, miss / T, adm / T


print("\n1. hit rate, admit on every miss (no admission cost)")
print("   slots  static |   LRU  | Belady")
for cap in (430, 472, 530):
    for ns in (250, 0):
        print(f"   {cap:5d}  {ns:6d} | {lru(cap, ns):.3f} | {belady(cap, ns):.3f}")

print("\n2. cost-aware admission, 430 slots: cost = CPU expert reads/token = misses + 2 x admissions")
rules = [("static set only, no admission", dict(gate="none"))]
rules += [(f"admit on >= {k} uses in last {w} tokens, LRU", dict(gate="freq", k=k, window=w))
          for k, w in ((2, 32), (3, 64), (4, 128), (6, 256))]
rules += [(f"ORACLE: admit if >= {r} uses in next {hz} tokens", dict(gate="oracle", r=r, horizon=hz, oracle_evict=True))
          for r, hz in ((2, 32), (3, 64), (4, 128))]
for ns in (250, 0):
    for name, kw in rules:
        hit, m, a = gated(430, ns, **kw)
        print(f"   static {ns:3d} | {name:46s} | hit {hit:.3f} | miss/tok {m:5.1f} | adm/tok {a:5.1f} | cost {m + 2 * a:6.1f}")

print("\n3. multi-token verify: unique experts per step of T tokens (1 token = 144)")
stat = set(static_gids)
for W in (1, 2, 3, 4, 6, 8):
    u = [len(set(np.array(toks[t:t + W]).reshape(-1).tolist())) for t in range(T - W + 1)]
    un = [len(set(np.array(toks[t:t + W]).reshape(-1).tolist()) - stat) for t in range(T - W + 1)]
    print(f"   T={W}: {np.mean(u):6.1f} experts ({np.mean(u) / 144:4.2f}x), non-static {np.mean(un):6.1f}, "
          f"best case (all drafts accepted) {W / (np.mean(u) / 144):4.2f}x tokens per expert read")
