"""Pack / unpack the E8M0 scale region of a gpt-oss-120B expert slot: the reference implementation of the packed
store layout (weight_repr "mxfp4_g32_ps4"). tools/pack_store.py converts whole stores with it; the kernels read the
result (kernels/gptoss_cpu_cap2.c, gptoss_cpu_multi.c, neural/q80/mxfp4_kernels.py); the tests and A/B tools use it as
the oracle. This script itself is a check of the format against a real store.

Raw slot (13,219,200 B, see kernels/gptoss_cpu_cap2.c):
    gate_codes [5760][90][16] | down_codes [2880][90][16] | gate_scales [5760][90] | down_scales [2880][90]
Packed slot (12,839,040 B) = the same 12,441,600 code bytes, then 8,640 scale rows of 46 B each
(gate rows then down rows, same order):
    byte 0      base  = min of the row's 90 scale bytes
    bytes 1..45 4-bit deltas d[g] = s[g] - base; byte 1+k = d[2k] | d[2k+1] << 4   (low nibble = even index)
Lossless iff every row's span (max - min) <= 15; pack_slot raises SpanError otherwise (it never clips).

    python tools\\scale_pack.py                       # synthetic + 3 real slots + a span scan of the real store
    python tools\\scale_pack.py --store D:\\other --experts 0,5,9 --scan-stride 16

Labels: every count/percentage printed by the real-store part is MEASURED on the store it is pointed at.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as NP                                                     # noqa: E402  (NEURAL_STORE_DIR)

H, GU, GK, RB = 2880, 5760, 90, 1440
OFF_DC = GU * RB                       # 8,294,400
OFF_GS = OFF_DC + H * RB               # 12,441,600
OFF_DS = OFF_GS + GU * GK              # 12,960,000
SLOT_RAW = OFF_DS + H * GK             # 13,219,200
NROWS = GU + H                         # 8,640 scale rows
SR_P4 = 1 + GK // 2                    # 46 B per packed row
SLOT_PACKED = OFF_GS + NROWS * SR_P4   # 12,839,040
assert SLOT_RAW == 13_219_200 and SLOT_PACKED == 12_839_040 and NROWS * GK == 777_600


class SpanError(ValueError):
    """A scale row spans more than 15 exponent steps: 4-bit deltas cannot store it."""
    def __init__(self, rows, spans):
        self.rows, self.spans = rows, spans
        super().__init__(f"{len(rows)} scale row(s) with span > 15 (rows {rows[:8].tolist()}..., spans {spans[:8].tolist()}...)")


def row_spans(raw):
    """(8640,) int array: max - min of each scale row of a raw slot (or of a [.., 8640*90] scale view)."""
    sc = np.asarray(raw, np.uint8)[..., OFF_GS:].reshape(-1, NROWS, GK)
    return sc.max(axis=-1).astype(np.int16) - sc.min(axis=-1).astype(np.int16)


def pack_scale_rows(sc, want_span=False):
    """Scale rows only: raw uint8 [n, 90] -> packed uint8 [n, 46] (byte 0 = row minimum, bytes 1..45 = nibble pairs,
    low nibble = even group). Raises SpanError if any row span > 15 (never clips). want_span: also return the
    per-row span (max - min) as uint8 [n]. This is the whole transform: codes are copied byte for byte."""
    sc = np.asarray(sc, np.uint8)
    assert sc.ndim == 2 and sc.shape[1] == GK, sc.shape
    base = sc.min(axis=1)
    delta = sc - base[:, None]                                   # uint8, >= 0 by construction
    span = delta.max(axis=1)
    bad = np.flatnonzero(span > 15)
    if bad.size:
        raise SpanError(bad, span[bad].astype(np.int16))
    rows = np.empty((sc.shape[0], SR_P4), np.uint8)
    rows[:, 0] = base
    rows[:, 1:] = delta[:, 0::2] | (delta[:, 1::2] << 4)
    return (rows, span) if want_span else rows


def unpack_scale_rows(rows):
    """packed uint8 [n, 46] -> raw uint8 [n, 90] (exact inverse of pack_scale_rows)."""
    rows = np.asarray(rows, np.uint8)
    assert rows.ndim == 2 and rows.shape[1] == SR_P4, rows.shape
    base = rows[:, 0:1]
    sc = np.empty((rows.shape[0], GK), np.uint8)
    sc[:, 0::2] = base + (rows[:, 1:] & 15)
    sc[:, 1::2] = base + (rows[:, 1:] >> 4)
    return sc


def pack_slot(raw):
    """raw uint8[13,219,200] -> packed uint8[12,839,040]. Raises SpanError if any row span > 15."""
    raw = np.asarray(raw, np.uint8)
    assert raw.shape == (SLOT_RAW,), raw.shape
    rows = pack_scale_rows(raw[OFF_GS:].reshape(NROWS, GK))
    out = np.empty(SLOT_PACKED, np.uint8)
    out[:OFF_GS] = raw[:OFF_GS]
    out[OFF_GS:] = rows.ravel()
    return out


def unpack_slot(packed):
    """packed uint8[12,839,040] -> raw uint8[13,219,200] (exact inverse of pack_slot)."""
    packed = np.asarray(packed, np.uint8)
    assert packed.shape == (SLOT_PACKED,), packed.shape
    sc = unpack_scale_rows(packed[OFF_GS:].reshape(NROWS, SR_P4))
    out = np.empty(SLOT_RAW, np.uint8)
    out[:OFF_GS] = packed[:OFF_GS]
    out[OFF_GS:] = sc.ravel()
    return out


def synthetic_slot(rng, span_mix=True):
    """A random raw slot whose scale rows all satisfy span <= 15 and exercise every nibble value:
    25% constant rows (span 0), 25% deltas in {0,1}, 50% deltas uniform 0..15 (span 15 almost surely).
    Bases in [105, 120] keep every scale in [105, 135], i.e. 2^(s-127) in [2^-22, 2^8]."""
    s = rng.integers(0, 256, size=SLOT_RAW, dtype=np.uint8)
    base = rng.integers(105, 121, size=(NROWS, 1)).astype(np.uint8)
    kind = rng.integers(0, 4, size=(NROWS, 1))
    d = rng.integers(0, 16, size=(NROWS, GK)).astype(np.uint8)
    if span_mix:
        d = np.where(kind == 0, 0, np.where(kind == 1, d & 1, d)).astype(np.uint8)
    s[OFF_GS:] = (base + d).ravel()
    return s


def _selftest_synthetic(n=6, seed=1):
    rng = np.random.default_rng(seed)
    for i in range(n):
        raw = synthetic_slot(rng)
        p = pack_slot(raw)
        assert p.shape == (SLOT_PACKED,)
        assert np.array_equal(p[:OFF_GS], raw[:OFF_GS])
        assert np.array_equal(unpack_slot(p), raw), f"synthetic slot {i}: unpack(pack(x)) != x"
    # edge rows: constant, span exactly 15, base 240 (top of the byte range with span 15)
    raw = synthetic_slot(rng)
    sc = raw[OFF_GS:].reshape(NROWS, GK)
    sc[0, :] = 0; sc[1, :] = 255; sc[2, :] = 240; sc[2, 17] = 255; sc[3, :] = 7; sc[3, ::2] = 22; sc[4, 89] = 200; sc[4, :89] = 185
    assert np.array_equal(unpack_slot(pack_slot(raw)), raw)
    # span 16 must be refused, not clipped
    sc[5, :] = 100; sc[5, 44] = 116
    try:
        pack_slot(raw); raise SystemExit("FAIL: span-16 row was packed")
    except SpanError as e:
        assert 5 in e.rows
    print(f"  (a) synthetic: {n} random slots + edge rows (constant / span 15 / base 240 / 0 and 255) round-trip exactly; span 16 is refused")


def _real(store, layer, experts):
    path = os.path.join(store, f"layer_{layer}.slots")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    assert mm.size % SLOT_RAW == 0, (path, mm.size)
    return mm.reshape(-1, SLOT_RAW)


def _hist(spans):
    n = spans.size
    edges = [("0", spans == 0), ("1-3", (spans >= 1) & (spans <= 3)), ("4-7", (spans >= 4) & (spans <= 7)),
             ("8-15", (spans >= 8) & (spans <= 15)), (">=16", spans >= 16)]
    return "  ".join(f"span {k}: {int(m.sum()):,} ({100 * m.sum() / n:.4f}%)" for k, m in edges)


def check_real(store, layer, experts):
    mm = _real(store, layer, experts)
    print(f"  (b) real slots from {os.path.join(store, f'layer_{layer}.slots')}: {mm.shape[0]} experts/layer")
    ok = True
    for e in experts:
        raw = np.array(mm[e])
        sp = row_spans(raw)
        big = np.flatnonzero(sp > 15)
        try:
            p = pack_slot(raw)
            rt = np.array_equal(unpack_slot(p), raw)
            print(f"      layer {layer} expert {e:3d}: spans max {int(sp.max())}, rows>7: {int((sp > 7).sum())}, rows>15: {big.size}; "
                  f"packed {p.size:,} B ({100 * (1 - p.size / SLOT_RAW):.2f}% smaller); unpack(pack(x)) == x: {rt}")
            ok &= rt
        except SpanError as ex:
            ok = False
            print(f"      layer {layer} expert {e:3d}: 4-BIT IMPOSSIBLE, rows with span > 15: {ex.rows.tolist()} spans {ex.spans.tolist()}")
    return ok


def scan(store, layers, stride, full_layers):
    """Span statistics over scale regions only (777,600 B per expert). Reads ~0.78 MB per sampled expert."""
    all_sp, per_slot_gt7, per_slot_gt15, nslots = [], [], [], 0
    bmin, bmax, smax = 255, 0, 0
    over = []                                            # (layer, expert, row, span)
    t0 = time.perf_counter(); nbytes = 0
    for L in layers:
        mm = _real(store, L, None)
        idx = range(mm.shape[0]) if L in full_layers else range(0, mm.shape[0], stride)
        for e in idx:
            sc = np.array(mm[e, OFF_GS:]).reshape(NROWS, GK); nbytes += sc.size
            mn, mx = sc.min(axis=1), sc.max(axis=1)
            sp = mx.astype(np.int16) - mn
            all_sp.append(sp)
            per_slot_gt7.append(int((sp > 7).sum())); per_slot_gt15.append(int((sp > 15).sum())); nslots += 1
            bmin, bmax, smax = min(bmin, int(mn.min())), max(bmax, int(mn.max())), max(smax, int(mx.max()))
            for r in np.flatnonzero(sp > 7):
                over.append((L, int(e), int(r), int(sp[r])))
    sp = np.concatenate(all_sp)
    print(f"  (c) span scan: {nslots} slots from {len(layers)} layers ({sp.size:,} scale rows, {nbytes / 1e6:.0f} MB read in {time.perf_counter() - t0:.1f} s)")
    print(f"      {_hist(sp)}")
    print(f"      max span {int(sp.max())}; rows > 15: {int((sp > 15).sum())} -> 4-bit deltas {'COVER EVERY SCANNED ROW' if sp.max() <= 15 else 'DO NOT COVER every row'}")
    print(f"      rows > 7: {int((sp > 7).sum())} in {int((np.array(per_slot_gt7) > 0).sum())} slots (max {max(per_slot_gt7)} in one slot); "
          f"row base min/max {bmin}/{bmax}, scale byte max {smax}")
    for o in over[:25]:
        print(f"        span>7: layer {o[0]} expert {o[1]} row {o[2]} ({'gate' if o[2] < GU else 'down'}) span {o[3]}")
    return int(sp.max()) <= 15


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=NP.STORE_DIR, help="a RAW store (weight_repr mxfp4_g32); default NEURAL_STORE_DIR")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", default="0,63,127", help="real slots to round-trip (layer --layer)")
    ap.add_argument("--scan-stride", type=int, default=32, help="span scan: every Nth expert of every layer (0 = skip the scan)")
    ap.add_argument("--scan-full-layer", type=int, default=0, help="also scan EVERY expert of this layer (-1 = none)")
    ap.add_argument("--no-real", action="store_true")
    A = ap.parse_args()
    print(f"scale_pack: raw slot {SLOT_RAW:,} B -> packed {SLOT_PACKED:,} B ({SLOT_RAW - SLOT_PACKED:,} B = {100 * (1 - SLOT_PACKED / SLOT_RAW):.2f}% less), "
          f"scale region {NROWS * GK:,} -> {NROWS * SR_P4:,} B")
    _selftest_synthetic()
    ok = True
    if not A.no_real:
        ok &= check_real(A.store, A.layer, [int(x) for x in A.experts.split(",")])
        if A.scan_stride:
            nl = len([f for f in os.listdir(A.store) if f.startswith("layer_") and f.endswith(".slots")])
            ok &= scan(A.store, list(range(nl)), A.scan_stride, {A.scan_full_layer} if A.scan_full_layer >= 0 else set())
    print("RESULT:", "OK" if ok else "FAILED (4-bit deltas do not cover every row checked)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
