"""Fit a linear "demand probe": final hidden state h_t -> which experts the NEXT H tokens will use.

    python tools/fit_demand_probe.py trace_a.json [trace_b.json ...] [--horizon 16] [--l2 1e-3]
    python tools/fit_demand_probe.py --synthetic 3000 --out /tmp/p.npz --pred-out /tmp/pred.npz

Input  (MEASURED): server `GET /neural/trace` JSON, either {"trace": {...}} or the dict itself, with
  "ids" [T][36][4] (required), "hidden" [T][2880] (required). Global expert id g = layer*128 + expert
  (4608 ids). Tokens are never mixed across files when building targets or past histograms.
Sample t: X_t = hidden[t], standardized with TRAIN mean/std; Y_t = experts used in tokens t+1..t+H
  (multi-hot over 4608, plus per-expert use counts for the hit-rate proxy). Only t with t+H < T is used.
Split: >1 file -> the last round(0.3*nfiles) files are test; 1 file -> first 70% / last 30% of samples,
  contiguous, with an H-sample purge gap so no train target window overlaps the test tokens.
Probe: sigmoid(X @ W + b), mean BCE per (token, expert) element + 0.5*l2*||W||^2 (bias not
  regularized), full-batch Adam, W=0 and b=logit(train freq) at init, so iteration 0 == freq baseline.
  NOTE: --l2 is relative to the per-element mean BCE, so 1e-3 is strong shrinkage; sweep 1e-5..1e-3.
  Loading big traces as JSON is memory-hungry (Python floats); hidden is cast to float32 on load.
Baselines on the test split (all rank the 4608 experts per token):
  probe      sigmoid scores (ranked on logits, no float saturation ties)
  past-H     counts of experts in tokens t-H+1..t (realizable, past-only)
  freq       train-set frequency of "expert used in next H" (token-independent)
  oracle     true future counts (ceiling for the metrics; not realizable)
Metrics (means over test tokens): AUC (tie-averaged ranks; positives = experts in Y), precision@64,
  precision@144, recall@K of the true set, and hit@K = fraction of the next-H expert USES (with
  multiplicity, 144/token) covered by the top-K predicted experts (K = --cache-k, default 430 = GPU
  slots). Ties are broken by a fixed random column permutation, identical for all predictors.
Labels: every output is SIMULATED (offline replay) on MEASURED traces. --synthetic fabricates tokens
  from an invented regime model: it is a plumbing test only and says nothing about the real model.
Outputs: --out npz (W [D,4608], bias, mean, std, horizon, meta_json); runtime scores =
  sigmoid(((h - mean)/std) @ W + b). --pred-out npz: test scores float16 [T_test,4608], token_index
  (within its file), file_index, horizon, meta_json. Cost: ~4*N*D*4608 flops per iteration.
"""
import argparse, json, os, sys, time
import numpy as np

NL, NX = 36, 128
E = NL * NX


def load_trace(path):
    d = json.load(open(path))
    tr = d.get("trace", d)
    if "ids" not in tr or "hidden" not in tr:
        sys.exit(f"{path}: needs 'ids' and 'hidden' (got {sorted(tr)}); re-record with hidden capture enabled")
    ids = np.asarray(tr["ids"], dtype=np.int64)
    hid = np.asarray(tr["hidden"], dtype=np.float32)
    if ids.ndim != 3 or ids.shape[1] != NL or len(hid) != len(ids) or hid.ndim != 2:
        sys.exit(f"{path}: bad shapes ids {ids.shape} hidden {hid.shape} (want [T,{NL},K] and [T,D])")
    return ids, hid


def build(ids, hid, H):
    """Per-file arrays: X [n,D], C [n,E] uint8 future use counts, P [n,E] uint8 past-H counts (n = T-H)."""
    T = len(ids)
    n = T - H
    if n < 1:
        sys.exit(f"file has {T} tokens, need > horizon {H}")
    U = np.zeros((T, E), np.uint8)
    g = (np.arange(NL)[None, :, None] * NX + ids).reshape(T, -1)
    ok = (ids >= 0).reshape(T, -1)
    U[np.repeat(np.arange(T), g.shape[1])[ok.ravel()], g.ravel()[ok.ravel()]] = 1
    C = np.zeros((n, E), np.uint8)
    for k in range(1, H + 1):
        C += U[k:k + n]
    P = np.zeros((T, E), np.uint8)
    for k in range(H):
        P[k:] += U[:T - k]
    return hid[:n], C, P[:n]


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def adam(p, g, m, v, s, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    """In-place Adam step (s = scratch, same shape as p)."""
    m *= b1
    m += np.multiply(g, 1 - b1, out=s)
    v *= b2
    v += np.multiply(np.multiply(g, g, out=s), 1 - b2, out=s)
    np.sqrt(v, out=s)
    s /= np.sqrt(1 - b2 ** t)
    s += eps
    np.divide(m, s, out=s)
    s *= lr / (1 - b1 ** t)
    p -= s


def passes(X, C, W, b, chunk, grad, want_loss):
    """One chunked pass over X: (sum BCE or None, sum_n x^T(p-y) or None, sum_n (p-y) or None)."""
    loss, gW, gb, tmp = 0.0, None, None, None
    if grad:
        gW, gb = np.empty_like(W), np.zeros_like(b)
        tmp = np.empty_like(W) if len(X) > chunk else None
    zb = np.empty((min(chunk, len(X)), E), np.float32)
    yb = np.empty_like(zb)
    for a in range(0, len(X), chunk):                                # preallocated buffers: no per-chunk malloc
        x = X[a:a + chunk]
        z, y = zb[:len(x)], yb[:len(x)]
        np.matmul(x, W, out=z)
        z += b
        np.copyto(y, C[a:a + chunk] > 0)
        if want_loss:                                                # softplus(z) - y*z, stable
            loss += float(np.logaddexp(0, z).sum(dtype=np.float64) - (y * z).sum(dtype=np.float64))
        if grad:
            z *= 0.5; np.tanh(z, out=z); z *= 0.5; z += 0.5          # z <- sigmoid(z)
            z -= y
            np.matmul(x.T, z, out=gW if a == 0 else tmp)
            if a:
                gW += tmp
            gb += z.sum(0)
    return loss, gW, gb


def fit(Xtr, Ctr, Xte, Cte, a):
    N, D = Xtr.shape
    f = np.clip((Ctr > 0).mean(0), 1e-4, 1 - 1e-4)
    W, b = np.zeros((D, E), np.float32), np.log(f / (1 - f)).astype(np.float32)
    mW, vW, sW, mb, vb, sb = (np.zeros_like(q) for q in (W, W, W, b, b, b))
    print(f"fit: N_train={N} N_test={len(Xte)} D={D} iters={a.iters} lr={a.lr} l2={a.l2} (BCE = mean per token-expert element)")
    for it in range(a.iters + 1):
        log = it % 50 == 0 or it == a.iters
        ltr, gW, gb = passes(Xtr, Ctr, W, b, a.chunk, it < a.iters, log)
        if log:
            lte = passes(Xte, Cte, W, b, a.chunk, False, True)[0]
            print(f"  it {it:4d}  train {ltr / (N * E):.5f}  test {lte / (len(Xte) * E):.5f}  |W| {np.linalg.norm(W):.2f}", flush=True)
        if it < a.iters:                                             # objective = mean BCE + 0.5*l2*||W||^2
            gW *= 1.0 / (N * E)
            gW += a.l2 * W
            adam(W, gW, mW, vW, sW, it + 1, a.lr)
            adam(b, gb / (N * E), mb, vb, sb, it + 1, a.lr)
    return W, b


def rank_metrics(S, C, perm, cache_k, chunk=512):
    """Mean per-token AUC, P@64, P@144, R@cache_k, hit@cache_k for score matrix S vs future counts C."""
    idx, out = np.arange(E), {k: [] for k in ("auc", "p64", "p144", "rec", "hit")}
    for a in range(0, len(S), chunk):
        s, c = S[a:a + chunk][:, perm], C[a:a + chunk][:, perm]
        o = np.argsort(s, axis=1, kind="stable")
        s, c = np.take_along_axis(s, o, 1), np.take_along_axis(c, o, 1)
        y = c > 0
        new = np.ones(s.shape, bool); new[:, 1:] = s[:, 1:] != s[:, :-1]
        end = np.ones(s.shape, bool); end[:, :-1] = new[:, 1:]
        st = np.maximum.accumulate(np.where(new, idx, 0), axis=1)
        en = np.minimum.accumulate(np.where(end, idx, E - 1)[:, ::-1], axis=1)[:, ::-1]
        npos = y.sum(1)
        u = (((st + en) / 2 + 1) * y).sum(1) - npos * (npos + 1) / 2
        v = (npos > 0) & (npos < E)
        out["auc"].append(u[v] / (npos[v] * (E - npos[v])))
        out["p64"].append(y[:, -64:].sum(1) / 64)
        out["p144"].append(y[:, -144:].sum(1) / 144)
        out["rec"].append(y[:, -cache_k:].sum(1) / np.maximum(npos, 1))
        out["hit"].append(c[:, -cache_k:].sum(1, dtype=np.int64) / np.maximum(c.sum(1, dtype=np.int64), 1))
    return {k: float(np.concatenate(v).mean()) for k, v in out.items()}


def synth(n, dim, rng, R=6, stay=0.95, sub=20, noise=1.0, snr=100.0):
    """Invented regime trace: R regimes, each prefers `sub` experts per layer; hidden = emb[regime] + noise."""
    pref = np.zeros((R, NL, NX), bool)
    for r in range(R):
        for l in range(NL):
            pref[r, l, rng.choice(NX, sub, replace=False)] = True
    logw = np.log(rng.lognormal(0, 0.5, (NL, NX)))[None] + np.where(pref, 0.0, np.log(0.03))
    reg = np.empty(n, np.int64)
    reg[0] = rng.integers(R)
    sw, nxt = rng.random(n) > stay, rng.integers(R, size=n)
    for t in range(1, n):
        reg[t] = nxt[t] if sw[t] else reg[t - 1]
    ids = np.empty((n, NL, 4), np.int16)
    for a in range(0, n, 1000):
        s = logw[reg[a:a + 1000]] - np.log(-np.log(rng.random((len(reg[a:a + 1000]), NL, NX), dtype=np.float32) + 1e-12))
        ids[a:a + 1000] = np.argpartition(-s, 3, axis=2)[:, :, :4]
    E_r = rng.standard_normal((R, dim)).astype(np.float32) * np.sqrt(snr / dim)   # ||emb||^2 = snr, per-dim noise 1
    h = E_r[reg] + noise * rng.standard_normal((n, dim), dtype=np.float32)
    h = h * rng.lognormal(0, 0.5, dim).astype(np.float32) + rng.standard_normal(dim).astype(np.float32) * 3
    return ids.astype(np.int64), h.astype(np.float16).astype(np.float32), reg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="trace JSON files (need ids + hidden)")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--chunk", type=int, default=4096, help="rows per matmul chunk")
    ap.add_argument("--cache-k", type=int, default=430, help="K for recall/hit-rate (GPU expert slots)")
    ap.add_argument("--train-frac", type=float, default=0.7, help="single-file temporal split")
    ap.add_argument("--out", default="reports/demand_probe.npz")
    ap.add_argument("--pred-out", default="reports/demand_probe_pred.npz")
    ap.add_argument("--synthetic", type=int, metavar="N", help="fabricate N tokens instead of reading traces")
    ap.add_argument("--synthetic-dim", type=int, default=384, help="hidden dim of synthetic data (real: 2880)")
    ap.add_argument("--synthetic-snr", type=float, default=100.0, help="synthetic ||regime embedding||^2 vs unit per-dim noise")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0, H, rng = time.time(), a.horizon, np.random.default_rng(a.seed)
    if a.synthetic:
        ids, hid, _ = synth(a.synthetic, a.synthetic_dim, rng, snr=a.synthetic_snr)
        traces, names = [(ids, hid)], [f"synthetic:{a.synthetic}"]
    elif a.paths:
        traces, names = [load_trace(p) for p in a.paths], a.paths
    else:
        ap.error("give trace JSON paths or --synthetic N")
    parts = [build(i, h, H) for i, h in traces]                       # per file: X, C, P (never crossing files)
    n_of = [len(p[0]) for p in parts]
    fidx = np.concatenate([np.full(n, k) for k, n in enumerate(n_of)])
    tidx = np.concatenate([np.arange(n) for n in n_of])
    X, C, P = (np.concatenate([p[j] for p in parts]) for j in range(3))
    if len(parts) > 1:
        nte = max(1, int(round(0.3 * len(parts))))
        tr_m, te_m = fidx < len(parts) - nte, fidx >= len(parts) - nte
        split = f"by file: {len(parts) - nte} train / {nte} test"
    else:
        ntr = int(a.train_frac * n_of[0])
        tr_m, te_m = tidx < ntr, tidx >= ntr + H
        split = f"temporal {a.train_frac:.0%}/{1 - a.train_frac:.0%} with {H}-sample purge gap"
    if not tr_m.any() or not te_m.any():
        sys.exit("empty train or test split; need more tokens/files")
    tr, te = np.flatnonzero(tr_m), np.flatnonzero(te_m)
    mean = X[tr].mean(0, dtype=np.float64).astype(np.float32)
    std = X[tr].std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    Xtr, Xte = (X[tr] - mean) / std, (X[te] - mean) / std
    Ctr, Cte, Pte = C[tr], C[te], P[te]
    del X
    print(f"[SIMULATED on {'SYNTHETIC' if a.synthetic else 'MEASURED'} traces] {names} horizon={H} split={split}")
    print(f"uses/token {Cte.sum() / len(Cte) / H:.1f}; distinct experts in next-{H} (test mean) {(Cte > 0).sum(1).mean():.0f}")
    W, b = fit(Xtr, Ctr, Xte, Cte, a)
    S = np.empty((len(Xte), E), np.float32)
    for i in range(0, len(Xte), a.chunk):
        S[i:i + a.chunk] = Xte[i:i + a.chunk] @ W + b                 # logits (sigmoid is monotone)
    freq = (Ctr > 0).mean(0).astype(np.float32)
    perm = np.random.default_rng(1234).permutation(E)
    rows = [("probe", S), (f"past-{H} hist", Pte), ("global freq", np.broadcast_to(freq, S.shape)),
            ("oracle (ceiling)", Cte)]
    res = {n: rank_metrics(s, Cte, perm, a.cache_k) for n, s in rows}
    K = a.cache_k
    print(f"\n{'predictor':<18}{'AUC':>8}{'P@64':>8}{'P@144':>8}{'R@' + str(K):>8}{'hit@' + str(K):>8}")
    for n, m in res.items():
        print(f"{n:<18}{m['auc']:8.4f}{m['p64']:8.4f}{m['p144']:8.4f}{m['rec']:8.4f}{m['hit']:8.4f}")
    meta = dict(label="SIMULATED on " + ("SYNTHETIC (plumbing test only)" if a.synthetic else "MEASURED traces"),
                files=names, horizon=H, l2=a.l2, lr=a.lr, iters=a.iters, n_train=len(tr), n_test=len(te),
                split=split, cache_k=K, metrics=res, dim=int(W.shape[0]), n_experts=E,
                apply="sigmoid(((h-mean)/std) @ W + bias)")
    for path in (a.out, a.pred_out):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mj = np.array(json.dumps(meta))
    np.savez(a.out, W=W, bias=b, mean=mean, std=std, horizon=np.int64(H), meta_json=mj)
    np.savez_compressed(a.pred_out, scores=sigmoid(S).astype(np.float16), token_index=tidx[te],
                        file_index=fidx[te].astype(np.int32), horizon=np.int64(H), meta_json=mj)
    print(f"\nsaved {a.out} and {a.pred_out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
