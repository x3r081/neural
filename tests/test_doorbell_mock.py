"""CPU-only mock of the --doorbell protocol. No GPU, no CUDA, no model, single thread.

It extracts the REAL decode_token / _decode_token_classic / _decode_token_bell / _book_tab / _copies_issue / _bell_abort /
_bell_recover / _bell_stats / apply_tab_updates / pick_victim / free_capbuf / refresh_point source out of server.py (AST) and
drives it against a scripted world whose "GPU" is a small stream-ordered simulator:

  * everything the host enqueues on the main stream (graph launches, slot-table updates, their pinned-staging H2D
    copies) sits in a FIFO and runs LATER, in order, when the simulated GPU gets time: never ("lazy": only when the
    host blocks), after every host operation ("eager"), or a random amount after every host operation ("random");
  * a doorbell graph blocks at its spin node until Doorbell's FLAG has reached its WANT (the exact int32 predicate
    of fused_core.k_bell), then k_fetch snapshots out_pin and the router writes pack_pin, from the slot table AS OF
    ITS POSITION in the stream;
  * when the host blocks (event sync) while the GPU is blocked at a spin, the spin "times out": FAILS is bumped and
    the graph runs on with whatever is in memory, like the real kernel after --doorbell-timeout-ms.

What it asserts, on every token, for seven scenarios x three GPU-progress modes:
  0. --doorbell 0 leaves the classic loop TEXTUALLY unchanged: _decode_token_classic == the committed server.py's
     decode_token (AST), and so are the helpers it uses (apply_tab_updates, pick_victim, ...).
  1. ... and operationally: BELL None executes EXACTLY the host operation sequence of the committed server.py (git HEAD;
     set DOORBELL_TEST_BASE=<rev> or file:<path> once this work is committed): same log (graph replays, event ops,
     stream waits, tab/copy enqueues, kernel calls), same kernel arguments, same state after every token. The
     doorbell build's own classic order (_decode_token_bell(early=False): a ring before each launch) issues the same
     operations too (per-kind subsequences), so the two copies of the loop cannot drift apart unnoticed.
  2. --doorbell 1 (early) issues the SAME operations (per-kind subsequences identical), with each replay of GA[L+1] between
     arm(seq) and ring(seq), the CPU kernel of layer L inside that window, every slot-table update / copy-stream
     fence / GPU-miss enqueue before the arm, every capture copy after the ring; sequence numbers
     token_index * 36 + L + 1; FLAG == WANT after every hand-off.
  3. Buffer ownership: what k_fetch reads is exactly what the CPU kernel of the previous layer wrote (or the zeros of a
     layer without misses); the GPU never overwrote pack_pin (the kernel's x) while the kernel ran; the pinned
     tab-update staging is never rewritten before its H2D copy ran; capture buffers are filled before their H2D copy.
  4. Results (token ids, kernel arguments, residency state) are identical to --doorbell 0.
  5. Mutations of the protocol (ring before the kernel, never ring, tab update after the launch, capture copy
     before the kernel) are DETECTED, so the assertions above are not vacuous.
  6. Recovery: a spin that gives up (host stalled in its CPU kernel, a lost ring, --doorbell-inject) raises DoorbellTimeout,
     the token is re-run in the classic order, and everything observable equals the run that never stalled; three strikes
     switch the early launch off (--doorbell-max-timeouts), 0 = never; a timeout inside the retry propagates.
  7. The per-token mode switch: the first --doorbell-warm-tokens tokens of a request and every token after one with more
     than --doorbell-faults-max page faults run the classic order, the others the early order (checked on the operation
     log, token by token), and the per-request stats (doorbell_tokens / _classic_tokens / _timeouts) count exactly that.

    python tests/test_doorbell_mock.py            # default: ~10 s of CPU (single thread)
    python tests/test_doorbell_mock.py --long     # 40 tokens per scenario/mode: run it in pieces, --only <scenario|unit|fallback|mutations|recovery|modeswitch>
    pytest tests/test_doorbell_mock.py
"""
import ast, collections, ctypes, os, random, subprocess, sys, time, types
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np


class NPT:
    """Just enough of a torch tensor for the mock (no torch import: a numpy array with .numpy(), views, numel(), dtype)."""

    def __init__(self, a):
        self.a = a
        self.dtype = types.SimpleNamespace(is_floating_point=bool(a.dtype.kind == "f"))

    def numpy(self):
        return self.a

    def numel(self):
        return self.a.size

    def __getitem__(self, sl):
        return NPT(self.a[sl])


def pinned(n, dt=np.int32):
    return NPT(np.zeros(n, dt))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import doorbell as DB                                                     # noqa: E402

REQUIRED = ("apply_tab_updates", "free_capbuf", "refresh_point", "victim_keys", "pick_victim", "admit_ok", "decode_token")
OPTIONAL = ("_decode_token_classic", "_decode_token_bell", "_book_tab", "_copies_issue", "_bell_abort", "_bell_recover",
            "_bell_stats")
WANT_ASSIGN = ()

NL, NE, K, H = 36, 128, 4, 2880
NM_OFF = H + 3 * K
PACKN = H + 4 * K
SLOTB = 13_219_200
UPD = 64
VP = ctypes.c_void_p


BASE_REV = os.environ.get("DOORBELL_TEST_BASE", "9dd3fe5")  # the server.py that --doorbell 0 must reproduce operation for operation:
                                                            # the last pre-doorbell main (the merge is e32cb5c); override with DOORBELL_TEST_BASE


def load_src(which, mutate=()):
    if which == "old" and BASE_REV.startswith("file:"):        # DOORBELL_TEST_BASE=file:<path to a pre-doorbell server.py>
        s = open(BASE_REV[5:], encoding="utf-8", newline="").read()
    elif which == "old":          # the committed (pre-doorbell) server.py
        s = subprocess.run(["git", "-C", REPO, "show", f"{BASE_REV}:server.py"], capture_output=True, check=True).stdout.decode("utf-8")
    else:
        s = open(os.path.join(REPO, "server.py"), encoding="utf-8", newline="").read()
    s = s.replace("\r\n", "\n")
    for old, new in mutate:
        assert s.count(old) == 1, (s.count(old), old)
        s = s.replace(old, new)
    return s


def extract(src):
    tree = ast.parse(src)
    funcs, assigns = [], []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in REQUIRED + OPTIONAL:
            funcs.append(n)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in WANT_ASSIGN:
            assigns.append(n)
    got = {f.name for f in funcs}
    assert set(REQUIRED) <= got, set(REQUIRED) - got
    return assigns, funcs


class Deadlock(AssertionError):
    pass


# ------------------------------------------------------------------------------------------------ simulated GPU
class GPU:
    """The main stream. Ops run in enqueue order; a doorbell graph can block the whole stream."""

    def __init__(self, W):
        self.W, self.q = W, collections.deque()
        self.tab = None                     # device slot table (np.int64)

    def enqueue(self, op):
        self.q.append(op)

    def run(self, limit=None):
        n = 0
        while self.q and (limit is None or n < limit):
            if not self.step():
                return False                # blocked at a spin
            n += 1
        return True

    def step(self, timeout_spin=False):
        W, op = self.W, self.q[0]
        kind = op[0]
        if kind == "graph":
            g = op[1]
            timed_out = False
            if g.bell and not DB.passed(int(W.bell_np[DB.FLAG]), int(W.bell_np[DB.WANT])):
                if not timeout_spin:
                    return False
                timed_out = True
                W.bell_np[DB.FAILS] += 1     # the spin gave up: the graph runs on with what is in memory
                W.bell_np[DB.POLLS] = 12345
                W.bell_np[DB.SPIN_US] = int(W.bell_np[DB.TIMEOUT_US])
                W.gave_up += 1
            elif g.bell:
                W.bell_np[DB.POLLS] = 7
                W.bell_np[DB.SPIN_US] = 3
            self.q.popleft()
            W.exec_graph(g, op[2], timed_out)
            return True
        self.q.popleft()
        if kind == "h2d":                   # pinned staging -> device buffer; the source is read NOW (execution time)
            _, view, src, snap = op
            cur = src.numpy().copy()
            W.check(np.array_equal(cur, snap), "tab-update staging was rewritten before its H2D copy executed")
            view.parent.data[:len(cur)] = cur
        elif kind == "tab":
            _, iv, vv = op
            n = iv.n
            self.tab[iv.parent.data[:n]] = vv.parent.data[:n]
        elif kind == "gadm":
            W.gadm_exec.append(op[1])
        return True

    def force_timeout(self):
        """The spin at the head of the stream gives up NOW (the host is stalled in its CPU kernel): the graph runs on with
        whatever is in memory. False when nothing is spinning (classic order: the flag is already there)."""
        if self.q and self.q[0][0] == "graph" and self.q[0][1].bell and not DB.passed(
                int(self.W.bell_np[DB.FLAG]), int(self.W.bell_np[DB.WANT])):
            self.step(timeout_spin=True)
            return True
        return False

    def drain_blocking(self):
        """The host blocks until the stream is empty (real time passes: a spin that nobody rings times out)."""
        while self.q:
            if not self.step(timeout_spin=True):
                raise AssertionError("unreachable")

    def run_until(self, pred):
        """The host blocks until pred() (an event) holds."""
        while not pred():
            if not self.q:
                raise Deadlock("host waits for an event nothing will complete")
            self.step(timeout_spin=True)


# ------------------------------------------------------------------------------------------------ fake CUDA objects
class FStream:
    def __init__(self, name, W):
        self.name, self.W = name, W

    def wait_stream(self, other):                       # torch: self.wait_event(other.record_event())  (a fresh event)
        self.W.log.append(("record", "<wait>", other.name))
        self.W.log.append(("wait_event", "<wait>", self.name))

    def wait_event(self, ev):
        self.W.log.append(("wait_event", ev.name, self.name))

    def synchronize(self):
        self.W.log.append(("stream_sync", self.name))
        self.W.gpu.drain_blocking()

    def query(self):
        self.W.log.append(("flush",))
        self.W.gpu.run()
        return not self.W.gpu.q


class FEvent:
    """CAPEV / ADMEV* / WAITEV: completion is a coin flip (as in the host-bench mock), not part of the GPU model."""

    def __init__(self, name, W, p_done=0.6):
        self.name, self.W, self.p = name, W, p_done

    def record(self, stream=None):
        self.W.log.append(("record", self.name, stream.name))

    def query(self):
        r = self.W.rng.random() < self.p
        self.W.log.append(("query", self.name, r))
        return r

    def synchronize(self):
        self.W.log.append(("ev_sync", self.name))


class FEvx:
    """EVX[L]: recorded by graph L on the simulated GPU."""

    def __init__(self, L, W):
        self.L, self.W, self.name, self.done = L, W, f"evx{L}", False

    def _seen(self):
        self.W.host_layer = self.L

    def synchronize(self):
        self.W.log.append(("ev_sync", self.name))
        self.W.gpu.run_until(lambda: self.done)
        self._seen()

    def query(self):
        self.W.log.append(("query", self.name, None))
        self.W.gpu.run()
        self.W.polls_left -= 1
        if not self.done and self.W.polls_left < 0:      # polling forever: real time passes, the spin times out
            self.W.gpu.run_until(lambda: self.done)
        if self.done:
            self._seen()
        return self.done


class FCtx:
    def __init__(self, stream, W):
        self.s, self.W = stream, W

    def __enter__(self):
        self.W.log.append(("ctx_enter", self.s.name))

    def __exit__(self, *a):
        self.W.log.append(("ctx_exit", self.s.name))


class FRow:
    def __init__(self, s, W):
        self.s, self.W = s, W

    def copy_(self, src, non_blocking=False):
        W = self.W
        W.log.append(("row_copy", self.s, src.tag))
        if src.tag < 100000:                              # a capture buffer: the CPU kernel of THIS layer must have filled it
            W.check(W.capb_written.get(src.tag) == (W.tokidx, W.host_layer), f"capture buffer {src.tag} copied before/without the kernel filled it")


class FBuf:
    def __init__(self, tag):
        self.tag = tag

    def data_ptr(self):
        return 0x7000_0000 + self.tag * SLOTB

    def copy_(self, *a, **k):
        pass

    def view(self, *a):
        return self


class FDevBuf:
    def __init__(self, name):
        self.name, self.data = name, np.zeros(UPD, np.int64)

    def __getitem__(self, sl):
        return FDevView(self, sl.stop)


class FDevView:
    def __init__(self, parent, n):
        self.parent, self.n = parent, n

    def copy_(self, src, non_blocking=False):
        W = self.parent.W
        W.log.append(("h2d_enqueue", self.parent.name, self.n))
        W.gpu.enqueue(("h2d", self, src, src.numpy().copy()))


class FTab:
    def __init__(self, W):
        self.W = W

    def index_copy_(self, dim, idx, val):
        self.W.log.append(("tab_enqueue", idx.n))
        self.W.gpu.enqueue(("tab", idx, val))


class FGraph:
    def __init__(self, W, L, bell):
        self.W, self.L = W, L
        self.fetch = L is None or L > 0            # k_fetch: every graph but GA[0]
        self.bell = bell and self.fetch            # ...preceded by the spin node when the doorbell is captured

    def replay(self):
        W = self.W
        if self.L is not None:
            W.ns["EVX"][self.L].done = False
        W.log.append(("replay", self.L))
        # the residency the host believes in at this call: every slot-table update it made BEFORE the launch must sit ahead of
        # this graph in the stream (checked when the router runs)
        W.gpu.enqueue(("graph", self, W.ns["TAB"].copy()))


class FTensor1:
    def __init__(self, name, W):
        self.name, self.W, self.v = name, W, None

    def __setitem__(self, i, v):
        self.v = v

    def copy_(self, src, non_blocking=False):
        self.W.log.append(("tcopy", self.name, getattr(src, "name", "?")))


class TLog(list):
    """Every host operation appends here; after each one the simulated GPU may make progress."""

    def __init__(self, W):
        super().__init__()
        self.W = W

    def append(self, x):
        super().append(x)
        self.W.tick()


# ------------------------------------------------------------------------------------------------ the world
class World:
    def __init__(self, which, variant, doorbell=None, progress="eager", seed=7, mutate=(), seq0=0):
        """doorbell: None = no spin nodes (classic), 'early' = --doorbell 1, 'fallback' = spin nodes but classic order."""
        self.which, self.progress, self.rng, self.seq0 = which, progress, random.Random(seed), seq0
        self.np_rng, self.grng = np.random.default_rng(seed), random.Random(seed + 99)
        self.log, self.kcalls = TLog(self), []
        self.gave_up = 0
        self.fetched, self.capb_written, self.kernel_ran = [], {}, set()
        self.router_tabs = []                       # per router execution: the device slot table it looked up
        self.gadm_exec = []
        self.host_layer, self.pack_owner, self.tokidx = None, None, 0
        self.polls_left = 50
        self.violations = []
        self.stall, self.stalled, self.tolerate = set(), [], False   # (token, layer) whose CPU kernel stalls until the spin gives up
        self.recovered = []
        self.garbage = None
        self.mid, self.tok_hash, self.seq_base = 0, 0, 0
        v = variant
        N_USABLE, STATIC_N = 300, 100
        ns = {"np": np, "time": time, "ctypes": ctypes, "H": H, "K": K, "NE": NE, "NL": NL, "NM_OFF": NM_OFF, "SLOTB": SLOTB, "UPD": UPD}
        self.ns = ns
        A = types.SimpleNamespace(wait=v.get("wait", "event"), threads=8, capbufs=16, admit_rule=v.get("admit_rule", "server"),
                                  victim=v.get("victim", "count"), admit_gpu_max=1, refresh_m=32, refresh_every=8, admit_window=4,
                                  admit_k=2, trace_max_tokens=10**6, doorbell_flush=v.get("flush", 1))
        ns["A"] = A
        main, copys = FStream("main", self), FStream("copy", self)
        self.main = main
        fake_cuda = types.SimpleNamespace(current_stream=lambda: main, stream=lambda s: FCtx(s, self),
                                          Event=lambda *a, **k: FEvent("<wait>", self), synchronize=lambda: self.gpu.drain_blocking())
        ns["torch"] = types.SimpleNamespace(cuda=fake_cuda)
        ns["COPYS"] = copys
        rr = self.np_rng
        GCc = rr.integers(0, 500, NL * NE).astype(np.int64)
        if v.get("hot"):                                  # the popular experts (low ids) of every layer are resident: layers with NO miss occur
            GCc = (NE - np.tile(np.arange(NE), NL)) * 10 + rr.integers(0, 3, NL * NE)
        res = np.argsort(-GCc)[:N_USABLE]
        TAB = np.full(NL * NE, -1, np.int64)
        TAB[res] = np.arange(N_USABLE)
        OWNER = np.full(N_USABLE, -1, np.int64)
        OWNER[TAB[res]] = res
        ns.update(TAB=TAB, OWNER=OWNER, GC=GCc, GEN=np.zeros(NL * NE, np.int64), PREF=np.zeros(NL * NE, np.int64), STATIC_N=STATIC_N,
                  N_USABLE=N_USABLE, BIG=np.int64(1) << 40, CAND=set(), PENDG=set(), PEND=[], RC={"captures": 0, "cand_marked": 0},
                  TOKI=[0], LAST=np.full(NL * NE, -1, np.int64), WCNT=np.zeros(NL * NE, np.int64), DEMAND=np.zeros(NL * NE, np.float32),
                  VMIN=[0.0], WHIST=collections.deque(), PROBE=None, TRACE=[], COUNTS=np.zeros((NL, NE), np.int64),
                  MODE={"counting": v.get("counting", False), "trace": v.get("trace", False), "trace_hidden": False, "ref": False},
                  NEAR_MISS=v.get("near_miss", False), NM_ON=v.get("near_miss", False), POLICY_STATE=v.get("policy", False),
                  GPU_MISS=v.get("gpu_miss", False), ADMIT_GPU=v.get("admit_gpu", False), ZC_MISSES=v.get("zc", 0),
                  GB=None, GA_REF=None, REF=None, ARENA_ROWS=set(), ADMI=[0], ZCI=[0], ADM_RING=64)
        ns["C"] = dict(cpu=0, hit=0, gpuadm=0, gpumiss=0, wait_s=0.0, cpu_s=0.0, replay_s=0.0, adm_s=0.0)
        self.gpu = GPU(self)
        self.gpu.tab = np.full(NL * NE, -1, np.int64)
        self.gpu.tab[res] = TAB[res]
        ns["slot_tab"] = FTab(self)
        ns["upd_idx_pin"] = pinned(UPD, np.int64); ns["upd_idx_np"] = ns["upd_idx_pin"].numpy()
        ns["upd_val_pin"] = pinned(UPD, np.int64); ns["upd_val_np"] = ns["upd_val_pin"].numpy()
        ns["upd_idx_dev"] = FDevBuf("idx"); ns["upd_idx_dev"].W = self
        ns["upd_val_dev"] = FDevBuf("val"); ns["upd_val_dev"].W = self
        ns["CAPB"] = [FBuf(i) for i in range(A.capbufs)]
        ns["CAPEV"] = [FEvent(f"cap{i}", self) for i in range(A.capbufs)]
        ns["CAPSTATE"] = [0] * A.capbufs
        ns["CP"] = (VP * K)(); ns["P"] = (VP * K)(); ns["BG"] = (VP * K)(); ns["BD"] = (VP * K)()
        ns["WV"] = np.zeros(K, np.float32)
        ns["ROWP"] = [0x10000000 + g * SLOTB for g in range(NL * NE)]
        ns["BGP"] = [0x60000000 + g * 23040 for g in range(NL * NE)]
        ns["BDP"] = [0x70000000 + g * 11520 for g in range(NL * NE)]
        ns["XP"], ns["OP"], ns["WP"], ns["SP"] = 1, 2, 3, 4
        ns["pack_np"] = np.zeros(PACKN, np.float32)
        ns["out_np"] = np.zeros((1, H), np.float32)
        ns["EVX"] = [FEvx(L, self) for L in range(NL)]
        bell = doorbell is not None
        self.driver_bell = doorbell == "early"      # decode_token(..., doorbell=True): the mode switch decides per token
        self.pf_script, self.pf_now = v.get("pf"), 0    # cumulative process page-fault counter at the start of token t
        ns["pmem"] = lambda: (self.pf_now, 0.0, 0.0)
        ns["GA"] = [FGraph(self, L, bell) for L in range(NL)]
        ns["GT"] = FGraph(self, None, bell)
        self.id_pin = [0]
        ns["id_pin"] = self.id_pin
        ns["cg"] = types.SimpleNamespace(pos_pin=FTensor1("pos_pin", self), pos=FTensor1("pos", self))
        ns["emb_pin"] = FTensor1("emb_pin", self)
        ns["mid"] = FTensor1("mid", self)
        ns["EMB_ROWS"] = lambda ids: FBuf(0)
        ns["rt"] = types.SimpleNamespace(_rows=[FRow(s, self) for s in range(N_USABLE)])
        ns["DoorbellTimeout"] = DB.DoorbellTimeout
        if bell:
            ns["BELL"] = DB.Doorbell(pinned(8), timeout_ms=DB.DEFAULT_TIMEOUT_MS,      # the server's default
                                     faults_max=v.get("faults_max", DB.DEFAULT_FAULTS_MAX), warm_tokens=v.get("warm", 0),
                                     max_timeouts=v.get("max_to", DB.DEFAULT_MAX_TIMEOUTS), inject=v.get("inject", 0))
            ns["BELL"].seq = seq0
            ns["BELL"].b[DB.FLAG] = ns["BELL"].b[DB.WANT] = DB.s32(seq0)
            ns["BELL"].pf_last = 0                      # a request that is already decoding: the fault baseline is known
            ns["BELL"].trace = lambda ev, seq: self.log.append((ev, seq))
            self.bell_np = ns["BELL"].b
        else:
            ns["BELL"] = None
            self.bell_np = np.zeros(8, np.int32)
        W = self

        class FLib:
            def gptoss_experts_cap(self_, E, P, XP, BG, BD, WP, OP, SP, thr, CP):
                W.kcalls.append((E, [P[i] for i in range(E)], [BG[i] for i in range(E)], [BD[i] for i in range(E)],
                                 ns["WV"][:E].tobytes(), [CP[i] for i in range(E)], XP, OP, WP, SP, thr))
                W.log.append(("kernel", E))
                L = W.host_layer
                if not W.tolerate:
                    W.check(W.pack_owner == (W.tokidx, L), f"kernel of layer {L} starts but pack_pin belongs to {W.pack_owner}")
                    W.check(bool(np.all(ns["pack_np"][:H] == float(W.tokidx * 100 + L + 1))), "the kernel's x (pack_pin[:H]) is not this layer's h_norm")
                mark = np.float32(W.tokidx * 100 + L + 1)
                ns["out_np"][:, :H // 2] = mark
                W.tick()
                if (W.tokidx, L) in W.stall and W.gpu.force_timeout():   # the host is stalled here (page faults): the spin gives up
                    W.tolerate = True                                    # ...and GA[L+1] runs on half-written out_pin / pack_pin
                    W.stalled.append((W.tokidx, L))
                if not W.tolerate:
                    W.check(W.pack_owner == (W.tokidx, L), "GPU wrote pack_pin while the CPU kernel was still reading it")
                ns["out_np"][:, H // 2:] = mark
                W.tick()
                for i in range(E):                    # capture buffers (copy-on-compute): filled by the kernel
                    if CP[i]:
                        W.capb_written[(CP[i] - 0x7000_0000) // SLOTB] = (W.tokidx, L)
                W.kernel_ran.add((W.tokidx, L))
                if not W.tolerate:
                    W.check(W.pack_owner == (W.tokidx, L), "GPU wrote pack_pin before the CPU kernel returned")
        ns["lib"] = FLib()
        if v.get("gpu_miss"):
            ns["ARENA_ROWS"].update(int(g) for g in rr.choice(NL * NE, 1500, replace=False) if TAB[g] < 0)
            ns["SCR_SLOTS"] = [N_USABLE - 1, N_USABLE - 2]
            ns["SLOT_FREE"] = [None] * N_USABLE
            ns["ADMEV"] = [FEvent(f"adm{i}", self, 0.5) for i in range(64)]
            ns["ADMEV2"] = [FEvent(f"adm2_{i}", self, 0.5) for i in range(64)]
            ns["ADM_PIN"] = [np.zeros(2, np.int64) for _ in range(64)]
            ns["ADM_WPIN"] = [np.zeros(1, np.float32) for _ in range(64)]

            class FAD:
                def __init__(self, name): self.name = name
                def copy_(self, src, non_blocking=False): W.log.append(("adm_copy", self.name, bytes(np.asarray(src).data)))
            ns["ADM_DEV"] = [FAD(f"dev{k}") for k in range(4)]
            ns["ADM_WDEV"] = [FAD(f"wdev{k}") for k in range(4)]

            class FG2:
                def __init__(self, k): self.k = k
                def replay(self):
                    W.log.append(("gadm_replay", self.k))
                    W.gpu.enqueue(("gadm", self.k))
            ns["GADM"] = [FG2(k) for k in range(4)]

            class FArena:
                def row_view(self_, g): return FBuf(100000 + g % 5000)
            ns["ARENA"] = FArena()
        # scripted routing
        self.route = []
        for t in range(v.get("tokens", 24)):
            lay = []
            for L in range(NL):
                p = 1.0 / (np.arange(NE) + 1.0) ** 0.8
                p /= p.sum()
                ids = self.np_rng.choice(NE, 8, replace=False, p=p)
                w = self.np_rng.random(K).astype(np.float32)
                lay.append(([int(x) for x in ids[:K]], w, [float(x) for x in ids[K:]]))
            self.route.append(lay)
        assigns, funcs = extract(load_src(which, mutate))
        mod = ast.Module(body=[*assigns, *funcs], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), f"<server.py {which}>", "exec"), ns)
        ns["GEN"][:] = rr.integers(0, 6, NL * NE)
        if not v.get("norefresh"):
            ns["refresh_point"](64)
        if "_bell_recover" in ns:                          # what the test harness must reset when the token is re-run
            orig = ns["_bell_recover"]

            def hooked(e, snap, orig=orig):
                self.garbage = (self.fetched[-1][1] if self.fetched else None, self.id_pin[0])   # what the failed attempt left behind
                self.fetched.clear(); self.kcalls.clear(); self.router_tabs.clear()
                self.kernel_ran = {k for k in self.kernel_ran if k[0] != self.tokidx}
                self.capb_written = {k: x for k, x in self.capb_written.items() if x[0] != self.tokidx}
                self.tolerate = False
                self.recovered.append(str(e))
                return orig(e, snap)
            ns["_bell_recover"] = hooked
        self.log.clear()

    # ---- simulated GPU progress
    def tick(self):
        if self.progress == "eager":
            self.gpu.run()
        elif self.progress == "random":
            self.gpu.run(limit=self.grng.randrange(0, 4))

    def check(self, ok, msg):
        if not ok:
            self.violations.append(msg)
            raise AssertionError(msg)

    def exec_graph(self, g, tab_at_launch, timed_out=False):
        ns = self.ns
        if g.L is not None:
            self.check(np.array_equal(self.gpu.tab, tab_at_launch),
                       f"router of layer {g.L} ran with a slot table older than what the host had enqueued before launching it")
        if g.fetch:                                       # k_fetch: c_out <- out_pin, read at THIS moment
            snap = ns["out_np"].copy()
            self.fetched.append((g.L, snap.tobytes()))
            prev = (NL - 1) if g.L is None else g.L - 1
            ran = (self.tokidx, prev) in self.kernel_ran
            want = np.full((1, H), np.float32(self.tokidx * 100 + prev + 1)) if ran else np.zeros((1, H), np.float32)
            if not timed_out:                             # a graph whose spin gave up runs on garbage by definition
                self.check(np.array_equal(snap, want), f"k_fetch of {g.L} read {'stale/partial' if ran else 'non-zero'} out_pin "
                           f"(layer {prev} kernel ran: {ran})")
            fsum = int(snap.astype(np.float64).sum())
        else:
            fsum = 0
        if g.L is None:                                   # tail: token id = f(final residual)
            self.id_pin[0] = (self.mid + fsum) % 1000
            return
        L, pk, route = g.L, ns["pack_np"], self.route[self.tokidx][g.L]
        self.router_tabs.append((L, hash(self.gpu.tab.tobytes())))
        pk[:H] = float(self.tokidx * 100 + L + 1)         # h_norm
        pk[H:H + K] = route[1]
        pk[H + K:H + 2 * K] = route[0]
        for j, e in enumerate(route[0]):                  # residency lookup: the device slot table AS OF this position in the stream
            pk[H + 2 * K + j] = 1.0 if int(self.gpu.tab[L * NE + e]) >= 0 else 0.0
        pk[NM_OFF:NM_OFF + K] = route[2]
        self.pack_owner = (self.tokidx, L)
        # residual stream: k_qkv READS mid (+ g_out + c_out), k_oproj OVERWRITES mid. Replaying a layer graph on its own
        # output therefore double-applies the layer: it is not idempotent.
        x = self.tok_hash if L == 0 else self.mid + fsum
        self.mid = (x * 31 + L + 7) & 0x7FFFFFFF
        ns["EVX"][L].done = True


def snapshot(W):
    W.gpu.drain_blocking()
    ns = W.ns
    return dict(GEN=ns["GEN"].copy(), TAB=ns["TAB"].copy(), OWNER=ns["OWNER"].copy(), CAND=sorted(ns["CAND"]), PENDG=sorted(ns["PENDG"]),
                PEND=list(ns["PEND"]), CAPSTATE=list(ns["CAPSTATE"]), slot_tab=W.gpu.tab.copy(), RC=dict(ns["RC"]),
                C={k: ns["C"][k] for k in ("cpu", "hit", "gpuadm", "gpumiss")}, LAST=ns["LAST"].copy(), COUNTS=ns["COUNTS"].copy(),
                WCNT=ns["WCNT"].copy(), TOKI=list(ns["TOKI"]),
                TRACE=[(a.tobytes(), b.tobytes(), None if c is None else c.tobytes()) for a, b, c, _ in ns["TRACE"]],
                ADMI=list(ns["ADMI"]), ZCI=list(ns["ZCI"]))


def same(a, b):
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            if not np.array_equal(va, vb):
                return k
        elif va != vb:
            return k
    return None


def step_token(W, t):
    W.tokidx = t
    W.log.clear()
    W.kcalls.clear()
    W.fetched.clear()
    W.router_tabs.clear()
    W.polls_left = 50
    W.tolerate = False
    W.tok_hash = (W.tok * 2654435761 + 12345) & 0x7FFFFFFF
    W.seq_base = W.ns["BELL"].seq if W.ns["BELL"] is not None else 0
    if W.pf_script:
        W.pf_now = W.pf_script[t]
    W.tok = W.ns["decode_token"](W.tok, 100 + t, True) if W.driver_bell else W.ns["decode_token"](W.tok, 100 + t)
    W.gpu.drain_blocking()


def by_kind(log):
    """The log grouped by operation kind, minus the doorbell's own steps and the poll count of a busy-wait (which
    depends on how far the simulated GPU had got)."""
    d = collections.defaultdict(list)
    for e in log:
        if e[0] in ("arm", "ring", "flush", "ring-lost") or (e[0] == "query" and str(e[1]).startswith("evx")):
            continue
        d[e[0]].append(e)
    return d


def check_bell_order(W, t, early, expect_flush, base=None):
    """Host-side protocol of one token of a doorbell world."""
    log = W.log
    arms = [(i, e[1]) for i, e in enumerate(log) if e[0] == "arm"]
    rings = [(i, e[1]) for i, e in enumerate(log) if e[0] == "ring"]
    assert len(arms) == len(rings) == NL, f"{len(arms)} arms / {len(rings)} rings in one token"
    for L in range(NL):                                              # sequence numbers: token_index * 36 + L + 1 (+ seq0, mod 2**32)
        want = ((W.seq0 + t * NL if base is None else base) + L + 1) & 0xFFFFFFFF
        assert arms[L][1] == rings[L][1] == want, (t, L, arms[L], rings[L], want)
        assert arms[L][0] < rings[L][0]
    replays = [(i, e[1]) for i, e in enumerate(log) if e[0] == "replay"]
    assert [r[1] for r in replays] == list(range(NL)) + [None]
    kernels = [i for i, e in enumerate(log) if e[0] == "kernel"]
    if not early:                                                    # classic order with spin graphs: ring, then launch, per graph
        for L in range(NL):
            a, r = arms[L][0], rings[L][0]
            assert r == a + 1 and log[r + 1] == ("replay", L + 1 if L + 1 < NL else None), (t, L)
            assert all(k < a for k in kernels if k < replays[L + 1][0] and k > replays[L][0])   # the layer's kernel ran before
        return
    assert replays[0][0] < arms[0][0]                                # GA[0]: no spin node, launched before the first hand-off
    for L in range(NL):
        a, r, nxt = arms[L][0], rings[L][0], replays[L + 1][0]
        assert a < nxt < r, (t, L)                                   # the launch of GA[L+1] (or the tail) sits inside arm..ring
        inside = {e[0] for e in log[a + 1:r]}
        assert inside <= {"replay", "kernel", "flush"}, (t, L, inside)   # ...and so does only the CPU kernel (+ the flush query)
        ks = [k for k in kernels if a < k < r]
        assert len(ks) <= 1 and all(nxt < k for k in ks), (t, L)     # the launch precedes the kernel
        assert (any(e[0] == "flush" for e in log[a + 1:r]) if expect_flush else True)
        # after the ring only the capture copies may follow, until the host blocks on the next router event: every other
        # enqueue the next graph must see (tab updates, the copy-stream fence, GPU-miss work) was done BEFORE the arm
        for e in log[r + 1:]:
            if e[0] == "ev_sync" and str(e[1]).startswith("evx") or e[0] in ("stream_sync",) or (e[0] == "query" and str(e[1]).startswith("evx")):
                break
            ok = e[0] in ("ctx_enter", "ctx_exit", "row_copy") or (e[0] == "record" and str(e[1]).startswith("cap")) or e[0] == "arm"
            assert ok, (t, L, "enqueue after the ring that the next graph should have seen", e)
    for i, e in enumerate(log):
        if e[0] == "row_copy" and e[2] < 100000:
            last_arm = max(a for a, _ in arms if a < i)
            last_ring = max(r for r, _ in rings if r < i)
            assert last_ring > last_arm, "capture copy issued between arm and ring (before the CPU kernel finished)"


def run(variant, name, progress, tokens, doorbell_mode="early", seq0=0):
    """Committed server (git HEAD) vs working-tree classic (BELL None) vs the doorbell build (early order through the driver, or
    its own classic order 'fallback'), every token."""
    Wo = World("old", variant, None, progress)
    Wc = World("new", variant, None, progress)
    Wb = World("new", variant, doorbell_mode, progress, seq0=seq0)
    for W in (Wo, Wc, Wb):
        W.tok = 5
    early = doorbell_mode == "early"
    stats = collections.Counter()
    for t in range(tokens):
        for W in (Wo, Wc, Wb):
            step_token(W, t)
        assert Wo.tok == Wc.tok == Wb.tok, (name, t, "token")
        # (1) --doorbell 0 == the committed server, operation for operation
        assert Wo.log == Wc.log, (name, t, next((i, x, y) for i, (x, y) in enumerate(zip(Wo.log, Wc.log)) if x != y)
                                  if len(Wo.log) == len(Wc.log) else (len(Wo.log), len(Wc.log)))
        assert Wo.kcalls == Wc.kcalls == Wb.kcalls, (name, t, "kernel args")
        assert Wo.fetched == Wc.fetched == Wb.fetched, (name, t, "what k_fetch read")
        assert Wo.router_tabs == Wc.router_tabs == Wb.router_tabs, (name, t, "the slot table each router looked up")
        # (2) --doorbell 1: the same operations, per kind in the same order (early AND the build's own classic order)
        ko, kb = by_kind(Wc.log), by_kind(Wb.log)
        assert ko.keys() == kb.keys(), (name, t, ko.keys() ^ kb.keys())
        for k in ko:
            assert ko[k] == kb[k], (name, t, k)
        check_bell_order(Wb, t, early, expect_flush=variant.get("flush", 1) == 1)
        assert Wb.gave_up == 0 and Wb.ns["BELL"].timeouts == 0
        assert int(Wb.bell_np[DB.FLAG]) == int(Wb.bell_np[DB.WANT]) == DB.s32(seq0 + (t + 1) * NL)
        # (4) identical state after the token
        s0, s1, s2 = snapshot(Wo), snapshot(Wc), snapshot(Wb)
        assert same(s0, s1) is None and same(s1, s2) is None, (name, t, same(s1, s2))
        stats["ops"] += len(Wb.log)
        stats["kernels"] += len(Wb.kcalls)
        stats["layers_without_miss"] += NL - len(Wb.kcalls)
        if (t + 1) % (4 if t < 32 else 8) == 0 and not variant.get("norefresh"):
            for W in (Wo, Wc, Wb):
                W.ns["refresh_point"]()
    bell = Wb.ns["BELL"]
    assert bell.req["tokens"] == (tokens if early else 0) and bell.req["timeouts"] == 0
    sn = snapshot(Wb)
    stats.update(captures=sn["RC"]["captures"], hit=sn["C"]["hit"], cpu=sn["C"]["cpu"], gpuadm=sn["C"]["gpuadm"], gpumiss=sn["C"]["gpumiss"])
    return stats


SCENARIOS = {
    "default": dict(),
    "poll": dict(wait="poll", modes=("random",)),
    "no-flush": dict(flush=0, modes=("random",)),
    "counting+trace+nearmiss": dict(counting=True, trace=True, near_miss=True),
    "policy(lru)": dict(policy=True, victim="lru"),
    "hot(resident, E==0 layers)": dict(hot=True, policy=True, victim="lru"),
    "gpu_miss+admit+zc": dict(gpu_miss=True, admit_gpu=True, zc=1, policy=True, victim="lru"),
}


# ------------------------------------------------------------------------------------------------ tests
def test_predicate_wraps():
    """Stale flags never satisfy a wait, across the int32 / uint32 wrap; open() releases any spin that starts."""
    for base in (0, 5, (1 << 31) - 3, (1 << 32) - 3):
        for d in range(-3, 4):
            want = (base + d) & 0xFFFFFFFF
            for lag in (1, 2, 36, 10**6):
                assert not DB.passed(DB.s32(want - lag), DB.s32(want))          # an older ring
            assert DB.passed(DB.s32(want), DB.s32(want))
            assert DB.passed(DB.s32(want + 1), DB.s32(want))                    # a later ring also releases (host never races ahead)
    b = DB.Doorbell(pinned(8), timeout_ms=1000)
    b.seq = (1 << 32) - 2
    b.b[DB.FLAG] = b.b[DB.WANT] = DB.s32(b.seq)                                # the state after a hand-off with that number
    seen = []
    for _ in range(5):
        seq = b.arm()
        assert not DB.passed(int(b.b[DB.FLAG]), int(b.b[DB.WANT]))              # armed, not rung: a spin would wait
        b.ring()
        assert DB.passed(int(b.b[DB.FLAG]), int(b.b[DB.WANT]))
        seen.append(seq)
    assert seen == [(1 << 32) - 1, 0, 1, 2, 3], seen
    b.arm()
    for bad, what in ((b.arm, "arm twice"),):
        try:
            bad()
        except AssertionError:
            pass
        else:
            raise SystemExit(what + " must assert")
    b.ring()
    try:
        b.ring()
    except AssertionError:
        pass
    else:
        raise SystemExit("ring without arm must assert")
    b.b[DB.FAILS] = 1
    assert b.read()[0] is True and b.read()[0] is False and b.timeouts == 1
    b.arm()                                                                    # open() is for quiet moments only
    try:
        b.open()
    except AssertionError:
        pass
    else:
        raise SystemExit("open() while armed must assert")
    b.ring()
    b.b[DB.WANT] = 77
    b.open()
    assert int(b.b[DB.FLAG]) == 77


def test_kernel_source_matches_predicate():
    """The mock GPU uses doorbell.passed(); the kernel must implement the same test and read the same words."""
    src = open(os.path.join(REPO, "fused_core.py"), encoding="utf-8").read()
    body = src[src.index("def k_bell"):src.index("BELL_POLL_CAP =")]
    assert "(flag - want) < 0" in body and "tl.load(bell + 1, volatile=True)" in body and "tl.load(bell, volatile=True)" in body
    assert "bell + 2" in body and "bell + 3" in body and "bell + 4" in body and "bell + 5" in body
    assert (DB.FLAG, DB.WANT, DB.TIMEOUT_US, DB.FAILS, DB.POLLS, DB.SPIN_US) == (0, 1, 2, 3, 4, 5)


def test_protocol(only=None, tokens=None, modes=("lazy", "eager", "random")):
    for name, v in SCENARIOS.items():
        if only and only != name:
            continue
        for progress in v.get("modes", modes):
            n = tokens or v.get("tokens", 8)
            r = run(dict(v, tokens=n), name, progress, n)
            print(f"  {name:26s} {progress:7s} {n:3d} tokens  {dict(r)}", flush=True)


def test_fallback_order_and_wrap():
    """Spin graphs with the classic order (ring right before each launch); sequence numbers crossing 2**31 and 2**32."""
    for seq0 in ((1 << 31) - 40, (1 << 32) - 40):
        r = run(dict(tokens=4), "fallback", "random", 4, doorbell_mode="fallback", seq0=seq0)
        print(f"  fallback order, seq0={seq0:#x}: {dict(r)}", flush=True)
        r = run(dict(tokens=4), "early-wrap", "eager", 4, doorbell_mode="early", seq0=seq0)
        print(f"  early order,    seq0={seq0:#x}: {dict(r)}", flush=True)


# server.py patches (exact substrings, each unique); every one breaks a rule of the protocol
_ARM_CTX = '            _tarm = time.perf_counter()\n            BELL.arm()                                # publish the sequence number this launch waits for'
_BOOK_PRE = '            newc = _book_tab(ids, base, cur, caps, upd, touched)\n            if newc:\n                COPYS.wait_stream(main)\n            C["adm_s"] += time.perf_counter() - _ta\n            _tl = _kd = 0.0'
_ISSUE_POST = '        if early:\n            if newc:\n                _copies_issue(newc)\n        else:'
_RING = 'BELL.ring()                       # the kernel ended with an sfence: out_pin is complete, release the spin'
MUTATIONS = {
    "ring before the CPU kernel": ([(_ARM_CTX, _ARM_CTX + "\n            BELL.ring()\n            BELL.armed = True\n            pass")], ("eager", "random")),
    "never ring": ([(_RING, "pass")], ("lazy", "eager", "random")),
    "tab update after the launch": ([(_BOOK_PRE, '            newc = []\n            C["adm_s"] += time.perf_counter() - _ta\n            _tl = _kd = 0.0'),
                                     (_ISSUE_POST, "        if early:\n            newc = _book_tab(ids, base, cur, caps, upd, touched)\n            if newc:\n                COPYS.wait_stream(main)\n"
                                                   "                _copies_issue(newc)\n        else:")], ("lazy", "eager", "random")),
    "capture copy before the kernel": ([(_BOOK_PRE, _BOOK_PRE.replace("                COPYS.wait_stream(main)\n", "                COPYS.wait_stream(main)\n                _copies_issue(newc)\n")),
                                        (_ISSUE_POST, "        if early:\n            pass\n        else:")], ("lazy", "eager", "random")),
}


def test_mutations_are_detected():
    v = dict(policy=True, victim="lru", tokens=12)      # captures happen (adaptive admission): tab updates and capture copies are exercised
    for name, (patch, modes) in MUTATIONS.items():
        for progress in modes:
            caught = None
            try:
                Wo = World("new", v, None, progress)
                Wm = World("new", v, "early", progress, mutate=patch)
                Wo.tok = Wm.tok = 5
                for t in range(v["tokens"]):
                    step_token(Wo, t)
                    step_token(Wm, t)
                    assert Wm.gave_up == 0, "a spin gave up in a fault-free run (the recovery would have hidden it)"
                    assert Wo.tok == Wm.tok, "token differs"
                    assert Wo.kcalls == Wm.kcalls, "kernel args differ (a different hit/miss split)"
                    assert Wo.fetched == Wm.fetched, "what k_fetch read differs"
                    assert Wo.router_tabs == Wm.router_tabs, "a router looked up a different slot table"
                    assert same(snapshot(Wo), snapshot(Wm)) is None, "state differs"
                    check_bell_order(Wm, t, True, True)
                    if (t + 1) % 4 == 0:
                        Wo.ns["refresh_point"]()
                        Wm.ns["refresh_point"]()
            except (AssertionError, DB.DoorbellTimeout) as e:      # noqa: PERF203
                caught = f"{type(e).__name__}: {e}"
            assert caught is not None, f"mutation '{name}' NOT detected in {progress} mode"
            print(f"  mutation {name!r:34s} {progress:7s} caught -> {caught[:105]}", flush=True)


def _pair(variant, progress, mutate=()):
    Wc = World("new", variant, None, progress)                       # the classic order, never stalls
    Wr = World("new", variant, "early", progress, mutate=mutate)
    Wc.tok = Wr.tok = 5
    return Wc, Wr


def _same_token(Wc, Wr, t, what):
    assert Wc.tok == Wr.tok, (what, t, "token id")
    assert Wc.kcalls == Wr.kcalls, (what, t, "kernel arguments of the final attempt")
    assert Wc.fetched == Wr.fetched and Wc.router_tabs == Wr.router_tabs, (what, t, "what k_fetch read / the tables the routers saw")
    d = same(snapshot(Wc), snapshot(Wr))
    assert d is None, (what, t, d)


def _stall_case(L, progress, mutate=(), tokens=4):
    """The CPU kernel of layer L of token 2 stalls until the spin of GA[L+1] (or the tail graph) gives up: that graph runs on a
    half-written out_pin and an overwritten pack_pin. The token must be re-run and everything observable must equal the run that
    never stalled: token ids, kernel arguments, what k_fetch read, the slot tables the routers saw, GEN / C / residency state."""
    v = dict(norefresh=True, tokens=tokens)
    Wc, Wr = _pair(v, progress, mutate)
    Wr.stall = {(2, L)}
    bell = Wr.ns["BELL"]
    for t in range(tokens):
        step_token(Wc, t)
        step_token(Wr, t)
        _same_token(Wc, Wr, t, f"stall at layer {L}, {progress}")
        if t == 2:
            assert Wr.stalled == [(2, L)], "the stall did not fire (no CPU misses at that layer?)"
            assert len(Wr.recovered) == 1 and bell.recoveries == 1 and bell.timeouts == 1 and bell.early is True
            msg = Wr.recovered[0]
            assert f"hand-off after layer {L}" in msg and "E=" in msg and "CPU kernel" in msg and "arm->ring" in msg, msg
            assert f"before layer {L + 1}" in msg
            raw, gid = Wr.garbage                                         # the failed attempt really produced garbage:
            assert len(np.unique(np.frombuffer(raw, np.float32))) >= 2, "the timed-out graph fetched a complete out_pin"
            if L == NL - 1:
                assert gid != Wc.tok, "the tail graph that gave up produced the right token by accident"
        elif t > 2:
            check_bell_order(Wr, t, True, True, base=Wr.seq_base)     # back to the early order right after the recovery
    assert not Wr.violations
    return Wr.recovered[0]


def test_stall_recovery_is_exact():
    for L in (0, 10, 34, 35):
        for progress in ("eager", "random", "lazy"):
            msg = _stall_case(L, progress)
        print(f"  stall in the CPU kernel of layer {L}: token re-run, ids/kernel args/k_fetch data/tables/state identical to the "
              f"no-stall run (eager, random, lazy)\n      log line: {msg}", flush=True)


def test_stall_recovery_adaptive_and_three_strikes():
    v = dict(policy=True, victim="lru", tokens=10)                    # captures / tab updates / PEND are live
    W = World("new", v, "early", "random")
    W.tok = 5
    W.stall = {(2, 5), (4, 20), (6, 33)}
    bell = W.ns["BELL"]
    for t in range(10):
        step_token(W, t)
        if (t + 1) % 4 == 0:
            W.ns["refresh_point"]()
    assert W.stalled == [(2, 5), (4, 20), (6, 33)], W.stalled
    assert bell.recoveries == 3 and bell.timeouts == 3 and bell.early is False and W.gave_up == 3 and not W.violations
    assert len(W.recovered) == 3
    print("  adaptive admission live, 3 stalls (tokens 2, 4, 6): every token completed, mock invariants held, early launch OFF after the "
          "3rd recovery, tokens 7-9 ran in the classic order with no further timeout", flush=True)


def test_second_timeout_in_the_classic_order_fails_the_request():
    Wc, W = _pair(dict(norefresh=True, tokens=4), "eager")
    W.stall = {(1, 10)}
    bell = W.ns["BELL"]
    step_token(W, 0)
    orig = W.ns["_bell_recover"]

    def lossy_after_recover(e, snap):
        r = orig(e, snap)

        def lost():                                                   # every flag store of the retry is lost
            bell.armed = False
        bell.ring = lost
        return r
    W.ns["_bell_recover"] = lossy_after_recover
    try:
        step_token(W, 1)
    except DB.DoorbellTimeout as e:
        print(f"  a timeout during the classic-order retry propagates (a bug, not a stall): {str(e)[:90]}", flush=True)
        return
    raise AssertionError("the retry's timeout was swallowed")


def test_lost_flag_store_is_recovered():
    """The flag store of layer 10 (token 1) is lost although the CPU output is complete: the spin gives up, the token is re-run."""
    Wc, W = _pair(dict(norefresh=True, tokens=5), "eager")
    bell = W.ns["BELL"]
    real_ring, hits = bell.ring, [0]

    def lossy_ring():
        hits[0] += 1
        if hits[0] == 36 + 11:
            bell.armed = False
            W.log.append(("ring-lost", bell.seq))
        else:
            real_ring()
    bell.ring = lossy_ring
    for t in range(5):
        step_token(Wc, t)
        step_token(W, t)
        _same_token(Wc, W, t, "lost flag store")
        if t >= 2:
            check_bell_order(W, t, True, True, base=W.seq_base)
    assert bell.recoveries == 1 and bell.early is True and W.gave_up == 1 and "before layer 11" in W.recovered[0]
    print(f"  lost flag store -> recovered, later tokens early again: {W.recovered[0][:120]}", flush=True)


def test_replaying_a_layer_graph_is_not_idempotent():
    """Why the fix re-runs the whole token: GA[L+1] READS mid and OVERWRITES it (fused_core.k_qkv reads it through _combine, k_oproj
    stores it), and b_body overwrites g_out, which k_qkv reads. A second replay of one layer graph on its own output double-applies
    the layer. (Model: the mock GPU's residual stream.)"""
    v = dict(norefresh=True, tokens=3)
    Wc, Wd = World("new", v, None, "eager"), World("new", v, None, "eager")
    Wc.tok = Wd.tok = 5
    g = Wd.ns["GA"][5]
    orig = g.replay
    g.replay = lambda: (orig(), orig())
    step_token(Wc, 0)
    step_token(Wd, 0)
    assert Wc.tok != Wd.tok, "a double replay of a layer graph must change the result"
    print("  replaying GA[5] twice changes the token id: layer graphs are not idempotent (mid / g_out are read and overwritten)", flush=True)


def test_default_timeout_is_the_tdr_safe_maximum():
    assert DB.MAX_TIMEOUT_MS == 1800 and DB.DEFAULT_TIMEOUT_MS == 800
    b = DB.Doorbell(pinned(8))
    assert int(b.b[DB.TIMEOUT_US]) == 800_000
    try:
        DB.Doorbell(pinned(8), timeout_ms=1801)
    except AssertionError:
        pass
    else:
        raise AssertionError("a timeout above the TDR-safe maximum must be refused")
    src = load_src("new")
    assert 'ap.add_argument("--doorbell-timeout-ms", type=int, default=800' in src and "1 <= A.doorbell_timeout_ms <= 1800" in src
    # the poll cap is the second bound (timer failure): ~1 s at the MEASURED 1.1 polls/us, i.e. also below the TDR
    assert 'ap.add_argument("--doorbell-poll-cap", type=int, default=1 << 20' in src and "1024 <= A.doorbell_poll_cap <= 1 << 20" in src
    fsrc = open(os.path.join(REPO, "fused_core.py"), encoding="utf-8").read()
    assert "BELL_POLL_CAP = 1 << 20" in fsrc and DB.MAX_POLL_CAP == 1 << 20
    assert DB.MAX_POLL_CAP / 1.1e6 < 1.5, "the poll cap must bound a spin well below the ~2 s TDR"


RECOVERY_MUTATIONS = {
    "GEN not restored": [("    GEN[:] = snap[0]\n", "    pass\n")],
    "C[cpu]/C[hit] not restored": [('    C["cpu"], C["hit"] = snap[1], snap[2]\n', "    pass\n")],
    "the retry keeps the early launch": [("    return _decode_token_bell(tok, pos, False)             # a timeout inside", "    return _decode_token_bell(tok, pos, True)             # a timeout inside")],
}


def test_recovery_mutations_are_detected():
    for name, patch in RECOVERY_MUTATIONS.items():
        caught = None
        try:
            _stall_case(10, "eager", mutate=patch)
        except (AssertionError, DB.DoorbellTimeout) as e:
            caught = f"{type(e).__name__}: {str(e)[:100]}"
        assert caught is not None, f"recovery mutation {name!r} NOT detected"
        print(f"  recovery mutation {name!r:36s} caught -> {caught}", flush=True)


def _fdef(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def test_classic_loop_is_textually_unchanged():
    """--doorbell 0 (BELL None) runs _decode_token_classic: the AST of the committed server.py's decode_token, unchanged, and
    every helper it calls is unchanged too. (test_protocol then checks the operation log of the two for equality.)"""
    old_src, new_src = load_src("old"), load_src("new")
    for name in ("apply_tab_updates", "free_capbuf", "refresh_point", "victim_keys", "pick_victim", "admit_ok", "pmem"):
        assert ast.dump(_fdef(old_src, name)) == ast.dump(_fdef(new_src, name)), f"{name} changed"
    new = _fdef(new_src, "_decode_token_classic")
    new.name = "decode_token"
    assert ast.dump(_fdef(old_src, "decode_token")) == ast.dump(new), "the classic loop is not the committed one"
    d = _fdef(new_src, "decode_token")                                # the dispatcher: BELL None goes straight to the classic loop
    first = d.body[1] if isinstance(d.body[0], ast.Expr) else d.body[0]
    assert " ".join(ast.unparse(first).split()) == "if BELL is None: return _decode_token_classic(tok, pos)", ast.unparse(first)
    print("  the classic loop and its helpers are AST-identical to the committed server.py; BELL None dispatches straight to it", flush=True)


def _bell(**kw):
    return DB.Doorbell(pinned(8), **kw)


def test_choose_unit():
    """The per-token mode switch: warm window, fault threshold (strictly greater), 32-bit wrap, baseline, off, new request."""
    b = _bell(faults_max=300, warm_tokens=3)
    faults = [10, 10, 10, 20, 1200, 900, 250, 20, 20, 300, 301, 5]            # faults DURING token t
    pf, got = 1000, []
    for f in faults:                                                           # choose() sees the cumulative count at the START of the token
        got.append(b.choose(pf))
        pf += f
    E, W_, F_ = (True, None), (False, "warm"), (False, "faults")
    assert got == [W_, W_, W_, E, E, F_, F_, E, E, E, E, F_], got
    assert b.req["tokens"] == 0 and b.req["classic"] == 6 and b.req["classic_warm"] == 3 and b.req["classic_faults"] == 3
    b = _bell(faults_max=300, warm_tokens=0)                                   # no warm window: the first token has no baseline yet
    assert b.choose(5) == W_ and b.choose(5) == E and b.faults_prev == 0
    assert b.choose(5 + 300) == E and b.choose(5 + 300 + 301) == F_ and b.choose(5 + 300 + 301 + 5) == E   # 300: early, 301: classic, 5: early
    b = _bell(faults_max=300, warm_tokens=0)                                   # the counter is 32 bit and wraps
    b.pf_last = (1 << 32) - 100
    assert b.choose(50) == E and b.faults_prev == 150
    b.pf_last = (1 << 32) - 100
    assert b.choose(250) == F_ and b.faults_prev == 350
    b = _bell(faults_max=300, warm_tokens=2)                                   # a new request restarts the warm window and the baseline
    for pf in (0, 0, 0, 0):
        b.choose(pf)
    assert b.req["classic"] == 2
    b.begin_request()
    assert b.n_req == 0 and b.pf_last is None and all(v == 0 for v in b.req.values())
    assert [b.choose(0)[1], b.choose(0)[1], b.choose(0)[0]] == ["warm", "warm", True]
    b.early = False                                                            # disabled after max timeouts: everything classic
    assert b.choose(0) == (False, "off") and b.req["classic_off"] == 1
    b = _bell(faults_max=0, warm_tokens=0)                                     # faults_max 0: any fault at all forces classic
    b.choose(0)
    assert b.choose(0) == E and b.choose(1) == F_ and b.choose(1) == E and b.choose(3) == F_
    print("  choose(): warm window, threshold (> faults_max), 32-bit wrap, baseline, off, begin_request: ok", flush=True)


def test_recovered_and_stats_unit():
    b = _bell(max_timeouts=2)
    assert b.recovered() is False and b.early and b.recovered() is True and not b.early and b.recoveries == 2
    assert b.req["classic"] == 2 and b.req["classic_timeout"] == 2
    b = _bell(max_timeouts=0)                                                  # 0 = never switch off
    for _ in range(10):
        assert b.recovered() is False
    assert b.early and b.recoveries == 10
    b = _bell(faults_max=300, warm_tokens=1)
    b.b[DB.FAILS] = 1
    b.b[DB.POLLS], b.b[DB.SPIN_US] = 1000, 2000
    assert b.read() == (True, 1000, 2000) and b.read() == (False, 1000, 2000) and b.timeouts == 1
    st = b.stats(4)
    need = {"doorbell", "doorbell_tokens", "doorbell_classic_tokens", "doorbell_timeouts"}
    assert need <= set(st), need - set(st)
    assert st["doorbell"] == 1 and st["doorbell_timeouts"] == 1 and st["doorbell_polls_per_tok"] == 500.0
    assert abs(st["doorbell_spin_ms_per_tok"] - 1.0) < 1e-9 and st["doorbell_disabled"] is False
    b.begin_request()
    assert b.stats(4)["doorbell_timeouts"] == 0 and b.timeouts == 1            # per request vs process lifetime
    for kw in (dict(faults_max=-1), dict(warm_tokens=-1), dict(max_timeouts=-1), dict(inject=-1), dict(timeout_ms=0)):
        try:
            _bell(**kw)
        except AssertionError:
            continue
        raise AssertionError(f"{kw} must be refused")
    print("  recovered() / max_timeouts (0 = never) / read() / stats() keys / argument validation: ok", flush=True)


def test_stats_without_doorbell():
    W = World("new", dict(norefresh=True, tokens=2), None, "eager")
    st = W.ns["_bell_stats"](7)
    assert st == {"doorbell": 0, "doorbell_tokens": 0, "doorbell_classic_tokens": 0, "doorbell_timeouts": 0}, st
    print("  --doorbell 0: the three stats keys are present and zero", flush=True)


def test_generate_wiring():
    """generate() (not extractable) is wired as designed: the serving decode passes doorbell=True, BELL.begin_request() follows
    reset(), the stats carry the doorbell keys; capture() opens the bell before its eager run; the fetch is the old one without BELL."""
    src = load_src("new")
    assert src.count("t = decode_token(t, P + n, True)") == 1
    assert src.count("t = decode_token(cur, Pn + i)") == 1                      # calibration / harness decode: classic order on spin graphs
    a = src.index("    t_prefill = time.perf_counter() - t0\n")
    b = src.index("    reset()\n    if BELL is not None:\n        BELL.begin_request()", a)   # the prefill timers' stats line may sit between
    assert b - a < 200, "BELL.begin_request() must follow the prefill timing and reset()"
    assert a < src.index("t = decode_token(t, P + n, True)")
    assert src.count("**_bell_stats(n_),") == 1
    assert "def capture(fn):\n    if BELL is not None:\n        BELL.open()" in src
    c = src[src.index("def _c_fetch():"):src.index("def a_full(L):")]
    assert "if BELL is not None:\n        bell_wait(BELL.pin, A.doorbell_poll_cap)\n    if ZC:\n        fetch(out_pin, c_out)" in c
    for name in ("a_full", "t_full", "t_full_sample"):
        assert "_c_fetch()" in ast.unparse(_fdef(src, name)), name
    print("  generate() / capture() / _c_fetch() wiring: ok", flush=True)


def test_mode_switch_in_the_driver():
    """decode_token(doorbell=True) picks the order of every token from the warm window and the previous token's faults: the
    operation log of each token has the early or the classic-on-spin-graphs shape accordingly, results equal the classic world's
    (adaptive admission live), and the request stats count exactly that."""
    faults = [10, 10, 10, 20, 1200, 900, 250, 20, 20, 300, 301, 5]
    pf, cum = 0, []
    for f in faults:
        cum.append(pf)
        pf += f
    n = len(faults)
    want = [t >= 3 and (t == 0 or faults[t - 1] <= 300) for t in range(n)]     # early?
    assert want == [False] * 3 + [True, True, False, False, True, True, True, True, False]
    v = dict(policy=True, victim="lru", tokens=n, warm=3, faults_max=300, pf=cum)
    for progress in ("eager", "random", "lazy"):
        Wc, Wr = World("new", v, None, progress), World("new", v, "early", progress)
        Wc.tok = Wr.tok = 5
        for t in range(n):
            step_token(Wc, t)
            step_token(Wr, t)
            _same_token(Wc, Wr, t, f"mode switch, {progress}")
            check_bell_order(Wr, t, want[t], True, base=Wr.seq_base)
            if (t + 1) % 4 == 0:
                Wc.ns["refresh_point"]()
                Wr.ns["refresh_point"]()
        bell = Wr.ns["BELL"]
        assert bell.req["tokens"] == sum(want) and bell.req["classic"] == n - sum(want)
        assert bell.req["classic_warm"] == 3 and bell.req["classic_faults"] == 3 and bell.req["timeouts"] == 0
        st = Wr.ns["_bell_stats"](n)
        assert st["doorbell_tokens"] + st["doorbell_classic_tokens"] == n and st["doorbell_timeouts"] == 0
        assert Wr.gave_up == 0 and bell.early
    print(f"  mode switch: warm 3 + faults > 300 -> classic ({n - sum(want)} of {n} tokens), the rest early; per-token log shapes, "
          f"results and stats as expected (eager, random, lazy)", flush=True)


def test_begin_request_restarts_warmup_in_the_driver():
    v = dict(norefresh=True, tokens=7, warm=2)
    Wc, Wr = World("new", v, None, "eager"), World("new", v, "early", "eager")
    Wc.tok = Wr.tok = 5
    want = [False, False, True, False, False, True, True]                      # a new request begins before token 3
    for t in range(7):
        if t == 3:
            Wr.ns["BELL"].begin_request()
        step_token(Wc, t)
        step_token(Wr, t)
        _same_token(Wc, Wr, t, "new request")
        check_bell_order(Wr, t, want[t], True, base=Wr.seq_base)
    print("  a new request restarts the warm window (tokens 3-4 classic again) and forgets the fault baseline", flush=True)


def test_mode_switch_and_stall_stats():
    """A stall in an early token: recovered, counted as a timeout and as a classic token, the other tokens unaffected."""
    v = dict(norefresh=True, tokens=6, warm=2)
    Wc, Wr = World("new", v, None, "eager"), World("new", v, "early", "eager")
    Wc.tok = Wr.tok = 5
    Wr.stall = {(3, 10)}
    for t in range(6):
        step_token(Wc, t)
        step_token(Wr, t)
        _same_token(Wc, Wr, t, "stall with mode switch")
    assert Wr.stalled == [(3, 10)] and len(Wr.recovered) == 1
    bell = Wr.ns["BELL"]
    r = bell.req
    assert (r["tokens"], r["classic"], r["classic_warm"], r["classic_timeout"], r["timeouts"]) == (3, 3, 2, 1, 1), r
    st = Wr.ns["_bell_stats"](6)
    assert (st["doorbell_tokens"], st["doorbell_classic_tokens"], st["doorbell_timeouts"]) == (3, 3, 1)
    assert bell.early and bell.recoveries == 1
    print("  stall + mode switch: stats = 3 early, 3 classic (2 warm + 1 retried), 1 timeout; results identical to the classic world", flush=True)


def test_inject_recovers_and_matches_classic():
    """--doorbell-inject N: every N-th early token loses one ring (after layer 17): the spin times out, the token is
    recovered, and everything observable equals the classic world (this is the switch for exercising the recovery on the GPU)."""
    v = dict(norefresh=True, tokens=8, inject=3, max_to=0)
    Wc, Wr = _pair(v, "eager")
    for t in range(8):
        step_token(Wc, t)
        step_token(Wr, t)
        _same_token(Wc, Wr, t, "inject")
        if t not in (2, 5) and t > 2:
            check_bell_order(Wr, t, True, True, base=Wr.seq_base)
    bell = Wr.ns["BELL"]
    assert Wr.gave_up == 2 and len(Wr.recovered) == 2 and bell.recoveries == 2 and bell.early
    assert all("hand-off after layer 17" in m and "before layer 18" in m for m in Wr.recovered), Wr.recovered
    r = bell.req
    assert (r["tokens"], r["classic"], r["classic_timeout"], r["timeouts"]) == (6, 2, 2, 2), r
    off = _bell(inject=0)
    assert [off.drop_layer() for _ in range(5)] == [-1] * 5
    on = _bell(inject=2)
    assert [on.drop_layer() for _ in range(6)] == [-1, DB.INJECT_LAYER, -1, DB.INJECT_LAYER, -1, DB.INJECT_LAYER]
    print("  --doorbell-inject 3: tokens 2 and 5 lost a ring after layer 17, both recovered, all 8 tokens identical to the classic world", flush=True)


def test_max_timeouts_zero_never_disables():
    v = dict(norefresh=True, tokens=6, max_to=0)
    Wc, Wr = _pair(v, "eager")
    Wr.stall = {(1, 10), (2, 12), (3, 14), (4, 16)}
    for t in range(6):
        step_token(Wc, t)
        step_token(Wr, t)
        _same_token(Wc, Wr, t, "max_to 0")
    bell = Wr.ns["BELL"]
    assert len(Wr.stalled) == 4 and bell.recoveries == 4 and bell.early, (Wr.stalled, bell.recoveries)
    check_bell_order(Wr, 5, True, True, base=Wr.seq_base)
    print("  --doorbell-max-timeouts 0: four stalls recovered, the early launch stays on", flush=True)


def test_mode_switch_mutations_are_detected():
    """The driver-level mode-switch test is not vacuous: a choose() that ignores the faults, the warm window, or uses >= is caught."""
    real = DB.Doorbell.choose

    def mk(ignore_faults=False, ignore_warm=False, ge=False):
        def choose(self, pf_now):
            n = self.n_req
            self.n_req += 1
            prev, self.pf_last = self.pf_last, pf_now
            self.faults_prev = None if prev is None else (pf_now - prev) & 0xFFFFFFFF
            if not self.early:
                why = "off"
            elif (n < self.warm_tokens and not ignore_warm) or self.faults_prev is None:
                why = "warm"
            elif not ignore_faults and (self.faults_prev >= self.faults_max if ge else self.faults_prev > self.faults_max):
                why = "faults"
            else:
                return True, None
            self.req["classic"] += 1
            self.req["classic_" + why] += 1
            return False, why
        return choose
    for name, kw in (("faults ignored", dict(ignore_faults=True)), ("warm window ignored", dict(ignore_warm=True)),
                     ("threshold >= instead of >", dict(ge=True))):
        DB.Doorbell.choose = mk(**kw)
        caught = None
        try:
            test_mode_switch_in_the_driver()
        except AssertionError as e:
            caught = f"AssertionError: {str(e)[:80]}"
        finally:
            DB.Doorbell.choose = real
        assert caught is not None, f"mode-switch mutation {name!r} NOT detected"
        print(f"  mode-switch mutation {name!r:28s} caught -> {caught}", flush=True)


def main(argv):
    long_ = "--long" in argv
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    t0 = time.perf_counter()
    if only in (None, "unit"):
        test_predicate_wraps()
        test_kernel_source_matches_predicate()
        test_classic_loop_is_textually_unchanged()
        test_choose_unit()
        test_recovered_and_stats_unit()
        test_stats_without_doorbell()
        test_generate_wiring()
        print("unit: ok", flush=True)
    if only in (None, "fallback"):
        test_fallback_order_and_wrap()
    if only in (None, "mutations"):
        test_mutations_are_detected()
    if only in (None, "recovery"):
        test_default_timeout_is_the_tdr_safe_maximum()
        test_replaying_a_layer_graph_is_not_idempotent()
        test_lost_flag_store_is_recovered()
        test_stall_recovery_is_exact()
        test_stall_recovery_adaptive_and_three_strikes()
        test_max_timeouts_zero_never_disables()
        test_inject_recovers_and_matches_classic()
        test_second_timeout_in_the_classic_order_fails_the_request()
        test_recovery_mutations_are_detected()
    if only in (None, "modeswitch"):
        test_mode_switch_in_the_driver()
        test_begin_request_restarts_warmup_in_the_driver()
        test_mode_switch_and_stall_stats()
        test_mode_switch_mutations_are_detected()
    if only is None or only in SCENARIOS:
        test_protocol(only=only if only in SCENARIOS else None, tokens=(40 if long_ else None))
    print(f"total {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
