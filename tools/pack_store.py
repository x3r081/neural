"""Convert a raw gpt-oss expert store into the PACKED-SCALE store (weight_repr "mxfp4_g32_ps4").

The raw store (tools/build_store.py) holds one file per layer, layer_<L>.slots, of n_experts rows of 13,219,200 B:
    gate_codes [5760][90][16] | down_codes [2880][90][16] | gate_scales [5760][90] | down_scales [2880][90]
The packed store keeps every code byte and stores each 90-byte E8M0 scale row as 46 B (byte 0 = the row's minimum, then
45 bytes of 4-bit deltas, low nibble = even group; tools/scale_pack.py), so a row is 12,839,040 B (-2.88%). The scale
is reconstructed as base + delta, exactly the original byte, so every kernel's arithmetic is unchanged. Lossless only
where every row's span (max - min) is <= 15: the converter REFUSES (exit 3) rather than clip, and --check-only proves
that for every slot of the source before anything is written.

  python tools\\pack_store.py --check-only                              # read-only span scan of all slots (go / no-go)
  python tools\\pack_store.py --dst <dir> --verify                        # convert (resumable), then verify
  python tools\\pack_store.py --dst <dir> --verify-only [--verify-mode sha]

Guarantees
  * The source store is opened read-only ('rb') and never written, renamed or deleted; --dst may not equal or contain it.
  * Streaming: one slot at a time through a 3-stage pipeline (read+hash | pack | write+hash), ~80 MB of RAM.
  * Resumable per layer: a layer is written to layer_<L>.slots.part, fsync'd, renamed, and recorded in pack_progress.json
    (which also pins the SHA-256 of the source metadata, so a changed source is detected; the file is deleted when the
    conversion finishes). metadata.json is written LAST, only when every layer is done: an unfinished destination
    cannot be opened by the runtime.
  * Every source layer's SHA-256 (of the raw stream, computed while reading) is checked against the SHA the source
    metadata recorded at build time; a mismatch aborts (--ignore-src-sha records it instead).
  * metadata.json carries the new descriptor (weight_repr mxfp4_g32_ps4, slot_bytes 12,839,040) and per-layer SHA-256
    of the packed files; expert_biases.pt is copied (hash-checked). The runtime compares weight_repr and slot_bytes
    strictly, so neither store is ever opened with the other's layout.
  * --verify: for every slot, unpack(dst) == src byte for byte AND pack(src) == dst (dst is canonical), the SHA-256 of the
    reconstructed raw stream == the source's recorded hash, the SHA-256 of each dst file == metadata.json, then the
    runtime's own Q80PrepackedStore.open() accepts dst with the packed layout and rejects it with the raw one.
    --verify-mode sha reads only dst (reconstructed-stream SHA vs the source's recorded SHA); full also reads the source.

Labels: every count, span and timing printed here is MEASURED on the stores it is pointed at. Runtime is I/O bound:
reads the whole source (~56.7 GiB) and writes ~55.1 GiB.
"""
import argparse, hashlib, json, os, queue, shutil, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import scale_pack as sp                                                     # noqa: E402
from neural.moe.layout import GPTOSS_LAYOUT, GPTOSS_LAYOUT_PS4, prepacked_descriptor   # noqa: E402  (torch-free)
import paths as NP                                                          # noqa: E402

FORMAT_MAGIC = "Q80-PREPACKED-SLOTS"       # == neural.q80.prepacked_store (that module imports torch; asserted in tests)
FORMAT_VERSION = 1
H, GU, GK = sp.H, sp.GU, sp.GK
OFF_GS, NROWS, SR_P4 = sp.OFF_GS, sp.NROWS, sp.SR_P4
SLOT_RAW, SLOT_PK = sp.SLOT_RAW, sp.SLOT_PACKED
assert SLOT_RAW == GPTOSS_LAYOUT.slot_bytes and SLOT_PK == GPTOSS_LAYOUT_PS4.slot_bytes
GIB = 1024 ** 3
PROGRESS = "pack_progress.json"
BIASES = "expert_biases.pt"


# ------------------------------------------------------------------ small helpers
class Fail(Exception):
    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def log(*a):
    print(*a, flush=True)


def fmt_s(s):
    s = int(round(s))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_exact(f, a):
    """Fill the writable buffer `a` from the binary file `f` (raises on a short file)."""
    mv = memoryview(a)
    n = 0
    while n < len(mv):
        k = f.readinto(mv[n:])
        if not k:
            raise Fail(f"unexpected end of file at byte {n} of a {len(mv):,} B read: {getattr(f, 'name', f)}")
        n += k


def write_all(f, a):
    mv = memoryview(a)
    while len(mv):
        k = f.write(mv)
        mv = mv[k:]


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_layers(spec, n):
    if spec in (None, "", "all"):
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    bad = [x for x in out if not 0 <= x < n]
    if bad:
        raise Fail(f"--layers {spec}: layer(s) {bad} outside 0..{n - 1}")
    return sorted(set(out))


def strip_row(d):
    return {k: v for k, v in d.items() if k != "row"}      # the human-readable label is informational (see open())


# ------------------------------------------------------------------ a 3-stage streaming pipeline
def run3(produce, transform, consume, depth=1):
    """produce() (thread) -> transform(item) (calling thread) -> consume(item) (thread); bounded queues of `depth`
    items, so at most ~depth+2 items exist per stage. Any exception stops every stage and is re-raised here."""
    q1, q2 = queue.Queue(depth), queue.Queue(depth)
    stop, errs = threading.Event(), []
    DONE = object()

    def put(q, x):
        while not stop.is_set():
            try:
                q.put(x, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def get(q):
        while not stop.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                pass
        return DONE

    def reader():
        try:
            for it in produce():
                if not put(q1, it):
                    return
            put(q1, DONE)
        except BaseException as e:                       # noqa: BLE001
            errs.append(e)
            stop.set()

    def writer():
        try:
            while True:
                it = get(q2)
                if it is DONE:
                    return
                consume(it)
        except BaseException as e:                       # noqa: BLE001
            errs.append(e)
            stop.set()

    tr, tw = threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)
    tr.start()
    tw.start()
    try:
        while True:
            it = get(q1)
            if it is DONE:
                break
            r = transform(it)
            if not put(q2, r):
                break
        put(q2, DONE)
    except BaseException as e:                           # noqa: BLE001
        errs.append(e)
        stop.set()
    tr.join()
    tw.join()
    if errs:
        raise errs[0]


# ------------------------------------------------------------------ source store
def load_src(src, need_layers=True):
    mp = os.path.join(src, "metadata.json")
    if not os.path.isfile(mp):
        raise Fail(f"{src}: no metadata.json (not a store, or unfinished)")
    raw_bytes = open(mp, "rb").read()
    meta = json.loads(raw_bytes.decode("utf-8"))
    if meta.get("magic") != FORMAT_MAGIC or meta.get("version") != FORMAT_VERSION:
        raise Fail(f"{src}: magic/version {meta.get('magic')!r}/{meta.get('version')!r} is not {FORMAT_MAGIC!r}/{FORMAT_VERSION}")
    lay = meta.get("layout") or {}
    if lay.get("weight_repr") != "mxfp4_g32" or lay.get("slot_bytes") != SLOT_RAW:
        raise Fail(f"{src}: layout weight_repr={lay.get('weight_repr')!r} slot_bytes={lay.get('slot_bytes')} - this converter "
                   f"reads only raw gpt-oss stores (mxfp4_g32, {SLOT_RAW:,} B/slot)")
    want = strip_row(prepacked_descriptor(GPTOSS_LAYOUT))
    have = strip_row(lay)
    diff = sorted(k for k in set(want) | set(have) if k not in ("n_layers", "n_experts") and want.get(k) != have.get(k))
    if diff:
        raise Fail(f"{src}: layout descriptor differs from the gpt-oss raw layout in {diff}")
    nl, ne = int(lay["n_layers"]), int(lay["n_experts"])
    if (nl, ne) != (GPTOSS_LAYOUT.n_layers, GPTOSS_LAYOUT.n_experts):
        log(f"WARNING: {src} has {nl} layers x {ne} experts (gpt-oss-120b is 36 x 128) - proceeding (test store?)")
    hashes = meta.get("sha256_per_layer") or {}
    if need_layers:
        for li in range(nl):
            p = os.path.join(src, f"layer_{li}.slots")
            if not os.path.isfile(p) or os.path.getsize(p) != ne * SLOT_RAW:
                raise Fail(f"{p}: missing or size != {ne} x {SLOT_RAW:,} B")
            if str(li) not in hashes:
                raise Fail(f"{mp}: no sha256_per_layer[{li}]")
    return meta, sha_bytes(raw_bytes), nl, ne


# ------------------------------------------------------------------ --check-only
def check_only(src, layers):
    meta, _, nl, ne = load_src(src)
    layers = parse_layers(layers, nl)
    log(f"check-only: scanning the scale region ({NROWS * GK:,} B) of {len(layers) * ne} slots of {src} (read-only; "
        f"~{len(layers) * ne * NROWS * GK / GIB:.1f} GiB)")
    hist = np.zeros(256, np.int64)
    over, slots_gt7 = [], 0
    bmin, bmax, smax = 255, 0, 0
    t0 = time.perf_counter()
    for li in layers:
        with open(os.path.join(src, f"layer_{li}.slots"), "rb", buffering=0) as f:
            def produce():
                for e in range(ne):
                    buf = np.empty(NROWS * GK, np.uint8)
                    f.seek(e * SLOT_RAW + OFF_GS)
                    read_exact(f, buf)
                    yield e, buf
            mx_l = 0

            def transform(it):
                e, buf = it
                sc = buf.reshape(NROWS, GK)
                mn, mx = sc.min(axis=1), sc.max(axis=1)
                return e, mn, mx, mx - mn

            def consume(it):
                nonlocal slots_gt7, bmin, bmax, smax, mx_l
                e, mn, mx, span = it
                hist[:] += np.bincount(span, minlength=256)
                bmin, bmax, smax = min(bmin, int(mn.min())), max(bmax, int(mn.max())), max(smax, int(mx.max()))
                mx_l = max(mx_l, int(span.max()))
                if (span > 7).any():
                    slots_gt7 += 1
                for r in np.flatnonzero(span > 15)[:8]:
                    over.append((li, e, int(r), int(span[r])))
            run3(produce, transform, consume, depth=4)
        el = time.perf_counter() - t0
        done = layers.index(li) + 1
        log(f"  layer {li:2d}: max span {mx_l:2d}   [{done}/{len(layers)} layers, {fmt_s(el)} elapsed]")
    n = int(hist.sum())
    mxs = int(np.flatnonzero(hist)[-1])
    log(f"scanned {len(layers) * ne} slots / {n:,} scale rows in {time.perf_counter() - t0:.1f} s (MEASURED)")
    log("  row span histogram: " + "  ".join(f"{k}:{int(hist[k]):,}" for k in range(mxs + 1) if hist[k]))
    log(f"  max span {mxs}; rows with span > 7: {int(hist[8:].sum()):,} (in {slots_gt7} slots); rows with span > 15: {int(hist[16:].sum()):,}; "
        f"row-min scale byte range {bmin}..{bmax}, max scale byte {smax}")
    if over:
        for o in over[:20]:
            log(f"  SPAN > 15: layer {o[0]} expert {o[1]} row {o[2]} ({'gate' if o[2] < GU else 'down'}) span {o[3]}")
        log("RESULT: 4-bit deltas DO NOT cover this store - do not convert (no partial/lossy mode exists)")
        return 3
    log(f"RESULT: OK - 4-bit deltas cover EVERY scale row of the {len(layers)} scanned layer(s)")
    return 0


# ------------------------------------------------------------------ conversion
class SpanFailure(Fail):
    def __init__(self, li, e, err):
        super().__init__(f"layer {li} expert {e}: {err} - 4-bit deltas cannot store this row (refusing to clip)", code=3)


def convert_layer(li, src_path, dst_part, ne, expect_sha, depth=1):
    """Stream one layer through the pipeline into dst_part. Returns (result dict, the still-OPEN output file): the
    caller flushes it to disk (fsync) and renames it - see Finisher, which overlaps that with the next layer."""
    st = {"max_span": 0, "rows_span_gt7": 0, "span_hist": np.zeros(16, np.int64)}
    h_raw, h_pk = hashlib.sha256(), hashlib.sha256()
    nbytes = [0]
    out = open(dst_part, "wb", buffering=0)

    def produce():
        with open(src_path, "rb", buffering=0) as f:
            for e in range(ne):
                buf = np.empty(SLOT_RAW, np.uint8)
                read_exact(f, buf)
                h_raw.update(buf)
                yield e, buf

    def transform(it):
        e, buf = it
        try:
            rows, span = sp.pack_scale_rows(buf[OFF_GS:].reshape(NROWS, GK), want_span=True)
        except sp.SpanError as ex:
            raise SpanFailure(li, e, ex)
        st["max_span"] = max(st["max_span"], int(span.max()))
        st["rows_span_gt7"] += int((span > 7).sum())
        st["span_hist"] += np.bincount(span, minlength=16)
        return e, buf, rows

    def consume(it):
        e, buf, rows = it
        codes = buf[:OFF_GS]
        flat = rows.reshape(-1)
        write_all(out, codes)
        write_all(out, flat)
        h_pk.update(codes)
        h_pk.update(flat)
        nbytes[0] += OFF_GS + flat.size

    try:
        run3(produce, transform, consume, depth)
        if nbytes[0] != ne * SLOT_PK:
            raise Fail(f"layer {li}: wrote {nbytes[0]:,} B, expected {ne * SLOT_PK:,}")
    except BaseException:
        out.close()
        raise
    return ({"sha256_raw": h_raw.hexdigest(), "sha256_packed": h_pk.hexdigest(), "bytes": nbytes[0],
             "max_span": st["max_span"], "rows_span_gt7": st["rows_span_gt7"], "span_hist": [int(v) for v in st["span_hist"]],
             "src_sha_ok": h_raw.hexdigest() == expect_sha}, out)


class Finisher:
    """One background thread that completes layers (fsync, close, rename, progress) while the main thread converts the
    next one, so the disk flush of layer N overlaps the CPU work of layer N+1. At most one job is pending (bounded
    dirty data); an exception in a job is re-raised in the caller at the next submit()/close()."""

    def __init__(self):
        self.q, self.err = queue.Queue(1), None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                job()
            except BaseException as e:                   # noqa: BLE001
                self.err = e
                return

    def _put(self, x):
        while self.t.is_alive():
            try:
                self.q.put(x, timeout=0.1)
                return
            except queue.Full:
                pass

    def submit(self, job):
        if self.err:
            raise self.err
        self._put(job)

    def close(self):
        self._put(None)
        self.t.join()
        if self.err:
            raise self.err


def convert(A):
    src, dst = os.path.realpath(A.src), os.path.realpath(A.dst)
    ns, nd = os.path.normcase(src), os.path.normcase(dst)            # Windows paths are case-insensitive
    if ns == nd or nd.startswith(ns + os.sep) or ns.startswith(nd + os.sep):
        raise Fail(f"--dst {dst} must be a different directory that neither contains nor lies inside --src {src}")
    meta, meta_sha, nl, ne = load_src(src)
    layers = parse_layers(A.layers, nl)
    os.makedirs(dst, exist_ok=True)
    mpath, ppath = os.path.join(dst, "metadata.json"), os.path.join(dst, PROGRESS)
    if os.path.exists(mpath):
        log(f"{dst} already holds a complete store (metadata.json present): nothing to convert"
            + (" (verifying)" if A.verify else " - use --verify-only to check it"))
        return
    if not os.path.isfile(os.path.join(src, BIASES)):
        raise Fail(f"{src}: no {BIASES}")
    prog = {"src": src, "src_metadata_sha256": meta_sha, "layers": {}}
    if os.path.exists(ppath):
        old = json.load(open(ppath, encoding="utf-8"))
        if old.get("src_metadata_sha256") != meta_sha:
            raise Fail(f"{ppath}: recorded for a different source metadata.json (sha {str(old.get('src_metadata_sha256'))[:12]} vs "
                       f"{meta_sha[:12]}): the source changed - use a fresh --dst")
        prog = old
    done = {int(k) for k, v in prog["layers"].items()
            if os.path.isfile(os.path.join(dst, f"layer_{k}.slots"))
            and os.path.getsize(os.path.join(dst, f"layer_{k}.slots")) == ne * SLOT_PK}
    todo = [li for li in layers if li not in done]
    all_todo = [li for li in range(nl) if li not in done]
    need = len(todo) * ne * SLOT_PK + (0 if os.path.exists(os.path.join(dst, BIASES)) else os.path.getsize(os.path.join(src, BIASES)))
    free = shutil.disk_usage(dst).free
    log(f"convert {src}\n     -> {dst}\n  {nl} layers x {ne} experts; {len(done)} layer(s) already done, {len(todo)} to do now; "
        f"reads {len(todo) * ne * SLOT_RAW / GIB:.1f} GiB, writes {need / GIB:.1f} GiB; free on the destination {free / GIB:.1f} GiB")
    if todo and free < need + int(A.min_free_gib * GIB):
        raise Fail(f"not enough free space on {dst}: need {need / GIB:.1f} GiB + {A.min_free_gib} GiB margin, have {free / GIB:.1f} GiB "
                   f"(use a --dst on a disk with more free space)")
    t0 = time.perf_counter()

    def finish(i, li, r, out, part, t1):
        """fsync + close + (source-SHA gate) + rename + progress for one layer; runs in the Finisher thread."""
        if not A.no_fsync:
            os.fsync(out.fileno())
        out.close()
        if not r["src_sha_ok"]:
            msg = (f"layer {li}: SHA-256 of the source rows ({r['sha256_raw'][:16]}...) != the value recorded in the source "
                   f"metadata.json ({meta['sha256_per_layer'][str(li)][:16]}...): the source layer differs from what was built")
            if not A.ignore_src_sha:
                os.replace(part, part.replace(".slots.part", ".slots.BAD-SOURCE-SHA"))
                raise Fail(msg + " - refusing (--ignore-src-sha to record it and go on)", code=4)
            log("WARNING: " + msg)
        os.replace(part, os.path.join(dst, f"layer_{li}.slots"))
        r["seconds"] = round(time.perf_counter() - t1, 2)
        prog["layers"][str(li)] = r
        atomic_json(ppath, prog)
        el = time.perf_counter() - t0
        rate = (i + 1) * ne * SLOT_RAW / el / GIB
        log(f"  layer {li:2d}: {r['seconds']:6.1f} s  max span {r['max_span']:2d}  src sha {'ok' if r['src_sha_ok'] else 'MISMATCH'}   "
            f"[{i + 1}/{len(todo)}; {rate:.2f} GiB/s read; elapsed {fmt_s(el)}, ETA {fmt_s(el / (i + 1) * (len(todo) - i - 1))}]")

    fin = Finisher()
    try:
        for i, li in enumerate(todo):
            part = os.path.join(dst, f"layer_{li}.slots.part")
            t1 = time.perf_counter()
            r, out = convert_layer(li, os.path.join(src, f"layer_{li}.slots"), part, ne, meta["sha256_per_layer"][str(li)], depth=A.depth)
            fin.submit(lambda i=i, li=li, r=r, out=out, part=part, t1=t1: finish(i, li, r, out, part, t1))
    finally:
        fin.close()        # completes the last layer even if a later one failed; re-raises a finisher error
    remaining = [li for li in all_todo if li not in layers]
    if remaining:
        log(f"partial run (--layers): layers {remaining[:8]}{'...' if len(remaining) > 8 else ''} are still to do; metadata.json not written")
        return
    # ---- expert biases (hash-checked copy), then metadata.json LAST
    bsrc, bdst = os.path.join(src, BIASES), os.path.join(dst, BIASES)
    if not os.path.exists(bdst):
        h1, h2 = hashlib.sha256(), hashlib.sha256()
        with open(bsrc, "rb") as f, open(bdst + ".part", "wb") as g:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h1.update(b)
                g.write(b)
            g.flush()
            os.fsync(g.fileno())
        os.replace(bdst + ".part", bdst)
        if sha_file(bdst) != h1.hexdigest():
            raise Fail(f"{BIASES} copy does not match its source")
    lay = prepacked_descriptor(GPTOSS_LAYOUT_PS4)
    lay["n_layers"], lay["n_experts"] = nl, ne
    L = prog["layers"]
    tot_hist = np.sum([L[str(li)]["span_hist"] for li in range(nl)], axis=0)
    newmeta = {"magic": FORMAT_MAGIC, "version": FORMAT_VERSION, "model_id": meta.get("model_id"), "layout": lay,
               "sha256_per_layer": {str(li): L[str(li)]["sha256_packed"] for li in range(nl)},
               "built_unix": int(time.time()), "build_seconds": round(time.perf_counter() - t0, 1),
               "conversion": ("scale packing of a raw gpt-oss store: every code byte copied exactly; each E8M0 scale row stored as "
                              "base (row minimum) + 4-bit deltas (46 B instead of 90); lossless, checked per layer against the source's "
                              "recorded SHA-256 of the raw stream; NO requantization"),
               "source": {"root": src, "metadata_sha256": meta_sha, "sha256_per_layer": meta["sha256_per_layer"],
                          "model_id": meta.get("model_id"), "built_unix": meta.get("built_unix")},
               "expert_biases_sha256": sha_file(bdst),
               "scale_packing": {"row_bytes": SR_P4, "rows_per_slot": NROWS, "max_row_span": int(np.flatnonzero(tot_hist)[-1]),
                                 "rows_span_gt7": int(sum(L[str(li)]["rows_span_gt7"] for li in range(nl))),
                                 "rows_total": int(tot_hist.sum()), "span_hist": [int(v) for v in tot_hist],
                                 "src_sha_mismatch_layers": [li for li in range(nl) if not L[str(li)]["src_sha_ok"]]}}
    atomic_json(mpath, newmeta)
    try:
        os.remove(ppath)                     # only meaningful while the conversion is unfinished
    except OSError:
        pass
    log(f"wrote {mpath}\nconversion done in {fmt_s(time.perf_counter() - t0)}: {nl * ne} slots, "
        f"{nl * ne * SLOT_PK / GIB:.1f} GiB packed (source {nl * ne * SLOT_RAW / GIB:.1f} GiB); max row span {newmeta['scale_packing']['max_row_span']}")


# ------------------------------------------------------------------ --verify
def verify(A):
    src, dst = os.path.realpath(A.src), os.path.realpath(A.dst)
    mp = os.path.join(dst, "metadata.json")
    if not os.path.isfile(mp):
        raise Fail(f"{dst}: no metadata.json - the conversion did not finish (re-run without --verify-only to resume)")
    dm = json.load(open(mp, encoding="utf-8"))
    lay = dm.get("layout") or {}
    nl, ne = int(lay.get("n_layers", 0)), int(lay.get("n_experts", 0))
    want = prepacked_descriptor(GPTOSS_LAYOUT_PS4)
    want["n_layers"], want["n_experts"] = nl, ne
    if dm.get("magic") != FORMAT_MAGIC or dm.get("version") != FORMAT_VERSION or strip_row(lay) != strip_row(want):
        raise Fail(f"{mp}: magic/version/layout descriptor is not the packed gpt-oss layout")
    full = A.verify_mode == "full"
    sm, _, snl, sne = load_src(src, need_layers=full)
    if (snl, sne) != (nl, ne):
        raise Fail(f"source is {snl} x {sne}, destination {nl} x {ne}")
    layers = parse_layers(A.layers, nl)
    log(f"verify ({A.verify_mode}): {len(layers)} layer(s) x {ne} experts of {dst}" + (f" against {src}" if full else f" against the SHA-256 the source recorded ({src}/metadata.json)"))
    bad = []
    t0 = time.perf_counter()
    nslots = 0
    for k, li in enumerate(layers):
        dp = os.path.join(dst, f"layer_{li}.slots")
        if not os.path.isfile(dp) or os.path.getsize(dp) != ne * SLOT_PK:
            bad.append((li, "missing or wrong size"))
            log(f"  layer {li:2d}: FAIL missing/size")
            continue
        h_dst, h_rec = hashlib.sha256(), hashlib.sha256()
        errs = []
        fs = open(os.path.join(src, f"layer_{li}.slots"), "rb", buffering=0) if full else None
        fd = open(dp, "rb", buffering=0)

        def produce():
            for e in range(ne):
                d = np.empty(SLOT_PK, np.uint8)
                read_exact(fd, d)
                h_dst.update(d)
                s = None
                if full:
                    s = np.empty(SLOT_RAW, np.uint8)
                    read_exact(fs, s)
                yield e, d, s

        def transform(it):
            e, d, s = it
            sc = sp.unpack_scale_rows(d[OFF_GS:].reshape(NROWS, SR_P4))
            if full:
                if not np.array_equal(d[:OFF_GS], s[:OFF_GS]):
                    errs.append((e, "codes differ"))
                ssc = s[OFF_GS:].reshape(NROWS, GK)
                if not np.array_equal(sc, ssc):
                    errs.append((e, "unpack(dst) scales != source scales"))
                try:
                    if not np.array_equal(sp.pack_scale_rows(ssc), d[OFF_GS:].reshape(NROWS, SR_P4)):
                        errs.append((e, "dst scale rows are not pack(source)"))
                except sp.SpanError:
                    errs.append((e, "source row span > 15"))
            return e, d, sc

        def consume(it):
            e, d, sc = it
            h_rec.update(d[:OFF_GS])
            h_rec.update(sc)

        try:
            run3(produce, transform, consume, depth=1)
        finally:
            fd.close()
            if fs:
                fs.close()
        probs = list(errs)
        if h_dst.hexdigest() != dm["sha256_per_layer"].get(str(li)):
            probs.append((-1, "SHA-256 of the dst file != metadata.json"))
        if h_rec.hexdigest() != sm["sha256_per_layer"].get(str(li)):
            probs.append((-1, "SHA-256 of unpack(dst) != the source's recorded SHA-256 of the raw rows"))
        nslots += ne
        if probs:
            bad.append((li, probs[:3]))
        el = time.perf_counter() - t0
        log(f"  layer {li:2d}: {'ok' if not probs else 'FAIL ' + str(probs[:3])}   [{k + 1}/{len(layers)}; elapsed {fmt_s(el)}, ETA {fmt_s(el / (k + 1) * (len(layers) - k - 1))}]")
    if (not bad and len(layers) == nl) or A.open_check:
        oc = open_check(dst, dm, nl, ne)
        if oc is not True:
            bad.append((-1, oc))
    if bad:
        log(f"VERIFY FAILED: {bad[:5]}")
        return 1
    log(f"VERIFY OK ({A.verify_mode}): {nslots} slots, {time.perf_counter() - t0:.1f} s (MEASURED)" +
        (": unpack(dst) == source byte for byte, dst == pack(source), file and stream SHA-256 match" if full else
         ": SHA-256 of the reconstructed raw stream == the source's recorded SHA-256, dst file SHA-256 == metadata.json"))
    return 0


def open_check(dst, dm, nl, ne):
    """The runtime's own store class must accept dst with the packed layout and refuse it with the raw one."""
    try:
        import dataclasses
        from neural.q80.prepacked_store import Q80PrepackedStore, FORMAT_MAGIC as M_, FORMAT_VERSION as V_
    except Exception as e:                                             # noqa: BLE001 (torch missing: report, do not fail)
        log(f"  open-check skipped ({type(e).__name__}: {e})")
        return True
    assert (M_, V_) == (FORMAT_MAGIC, FORMAT_VERSION)
    pk = dataclasses.replace(GPTOSS_LAYOUT_PS4, n_layers=nl, n_experts=ne)
    raw = dataclasses.replace(GPTOSS_LAYOUT, n_layers=nl, n_experts=ne)
    st = Q80PrepackedStore(dst, layout=pk).open(expect_model_id=dm.get("model_id"), shared=False)
    if st.get("status") != "ok":
        return f"runtime open() REJECTED the packed store: {st}"
    st2 = Q80PrepackedStore(dst, layout=raw).open(expect_model_id=dm.get("model_id"), shared=False)
    if st2.get("status") == "ok":
        return "runtime open() ACCEPTED the packed store with the RAW layout (must be refused)"
    log(f"  open-check: runtime open() accepts it as {pk.weight_repr} ({st['size_gib']} GiB) and refuses the raw layout "
        f"({st2.get('reason')}; differs {st2.get('differs')})")
    return True


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=NP.STORE_DIR, help="raw store (read-only; default NEURAL_STORE_DIR / paths.py)")
    ap.add_argument("--dst", default=None, help="packed store directory to create / resume / verify")
    ap.add_argument("--check-only", action="store_true", help="read-only: scan every source slot's scale rows for span > 15; write nothing")
    ap.add_argument("--verify", action="store_true", help="after converting, verify the destination")
    ap.add_argument("--verify-only", action="store_true", help="only verify an existing destination")
    ap.add_argument("--verify-mode", choices=("full", "sha"), default="full",
                    help="full: byte-for-byte vs the source (reads both); sha: reads only dst, compares the reconstructed stream's SHA-256 "
                         "with the SHA the source metadata recorded")
    ap.add_argument("--open-check", action="store_true", help="also run the runtime open() acceptance check on a partial --layers verify")
    ap.add_argument("--layers", default="all", help="e.g. 0,5-9 (convert / verify / check only these; metadata.json is written once all are done)")
    ap.add_argument("--ignore-src-sha", action="store_true", help="do not abort when a source layer's SHA-256 differs from its recorded one")
    ap.add_argument("--no-fsync", action="store_true")
    ap.add_argument("--depth", type=int, default=1, help="pipeline queue depth (RAM ~ (depth+2) x 26 MB)")
    ap.add_argument("--min-free-gib", type=float, default=1.0)
    A = ap.parse_args()
    try:
        if A.check_only:
            return check_only(os.path.realpath(A.src), A.layers)
        if not A.dst:
            raise Fail("--dst is required (or use --check-only)")
        if not A.verify_only:
            convert(A)
        if A.verify or A.verify_only:
            return verify(A)
        return 0
    except Fail as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
