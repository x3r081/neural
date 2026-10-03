"""Doorbell: host <-> GPU hand-off word for `server.py --doorbell 1`.

WHY.  Per layer the host used to (1) run the CPU expert kernel, (2) do slot-table bookkeeping, (3) call
cudaGraphLaunch for the next layer's graph, and only then did the GPU restart: the GPU idled for (2) + (3) +
the launch latency, 36 times per token. With a doorbell the next graph is launched BEFORE the CPU kernel;
its first node (fused_core.k_bell) spins on a word in pinned host memory until the host "rings" it after
the CPU kernel's output is complete, so the GPU resumes a few microseconds after the ring, with no host
round trip. Nothing else changes: the same kernels run in the same stream order on the same data.

WORD LAYOUT (int32 x 8, pinned host memory, mapped into the GPU's address space like out_pin / pack_pin):

    [0] FLAG        host -> GPU   sequence number of the last ring
    [1] WANT        host -> GPU   sequence number the graph about to run (or running) waits for
    [2] TIMEOUT_US  host -> GPU   the spin gives up after this many microseconds (globaltimer)
    [3] FAILS       GPU -> host   count of spins that gave up (never reset; the host compares)
    [4] POLLS       GPU -> host   iterations of the last spin
    [5] SPIN_US     GPU -> host   duration of the last spin, microseconds (telemetry)

PROTOCOL.  A graph's captured kernel arguments are FIXED, so the value a spin waits for cannot be a kernel
argument: the host publishes it in WANT before every launch and the kernel reads it once at its start.
The k-th doorbell of the process (k = 1, 2, ...; one per CPU-kernel -> next-graph hand-off, 36 per token,
so k = token_index * 36 + L + 1 for the hand-off after layer L) is:

    host:  arm()   WANT = k        (before the graph launch; the previous spin has finished: the host has
                                    waited for that graph's router event)
           ... launch graph, run CPU kernel L (ends with _mm_sfence), write out_pin ...
           ring()  FLAG = k        (a plain store after the sfence: see MEMORY ORDER)
    GPU:   spins until (int32)(FLAG - WANT) >= 0, then the graph continues.

Sequence numbers are 32-bit, wrap modulo 2**32 and are compared as a signed difference, so no reset is ever
needed and any stale FLAG (older ring) is < WANT for 2**31 doorbells. FLAG is only ever written by ring() / open().

MEMORY ORDER (why the GPU sees the CPU kernel's output when it sees the flag; the doorbell changes no computation).
  1. Every store the kernel makes to out_pin is either an ordinary store or (the capture copies) a non-temporal
     one; each OpenMP worker executes _mm_sfence() after its last store (kernels/gptoss_cpu_cap2.c) and the
     parallel region's closing barrier is crossed by all of them, so when gptoss_experts_cap() returns to Python
     every worker's stores to out_pin are globally visible, and the sfence has drained the non-temporal ones.
  2. ring() is a plain aligned 4-byte store to FLAG issued afterwards by the host thread. x86 keeps stores in
     program order (TSO) and Python / ctypes calls are compiler barriers, so no observer can see the new FLAG
     before the out_pin data (PCIe reads of pinned host memory are cache-coherent on x86).
  3. On the GPU the flag is read with ld.volatile (re-fetched from host memory every iteration, never cached or
     hoisted). The data behind the flag is read by the NEXT kernel of the graph, k_fetch, not by the spin kernel:
     the kernel boundary orders that read after the spin's exit, exactly as it was ordered after the host's
     cudaGraphLaunch before. tools/doorbell_selftest.py stage 4 checks this chain on the GPU (fresh random data
     written into the pinned source after the launch and before the ring, 2000 replays, every one must be seen).
  4. The other direction (GPU -> host) is the ordinary one: FAILS / POLLS / SPIN_US are stored by the spin kernel
     and read by the host only after the router event of the same graph, i.e. after later kernels of that graph.
  Host stores that the graph must see (slot-table updates, the copy-stream fence, GPU-miss work) are enqueued on
  the main stream BEFORE the early launch (server._decode_token_bell), so stream order carries them.

TIMEOUT AND RECOVERY.  WDDM's TDR kills a kernel that runs ~2 s. The spin gives up after TIMEOUT_US (default 800 ms,
maximum 1.8 s) or after POLL_CAP polls (a second bound in case the timer misbehaves; MEASURED rate 1.1 polls/us, so the
default cap of 2**20 is ~1 s), bumps FAILS and lets the graph run on with stale data. That cannot fault (the graph only
computes floats and indexes with router ids < 128), it just produces garbage. The host reads FAILS after every router event
(server._decode_token_bell), calls torch.cuda.synchronize() and raises DoorbellTimeout; server.decode_token catches it and
RE-RUNS THE WHOLE TOKEN in the classic order (a ring before every launch, so every spin exits at once). Why the whole token and
not just layer L+1:
  * GA[L+1] READS mid and g_out (k_qkv, through mid + g_out + c_out) and OVERWRITES both (k_oproj stores mid, b_body stores
    g_out), so replaying it a second time is not idempotent, and nothing saved their pre-images (the mock test's
    test_replaying_a_layer_graph_is_not_idempotent shows the double application). A layer re-run needs those pre-images:
    saving them costs extra nodes in every graph (or a copy inside k_bell), a hot-path change that this port cannot validate
    without the GPU, for an event that is rare by construction.
  * The token's inputs are all on the host: (tok, pos) and the K/V cache below pos. Everything the failed attempt wrote is
    per-token output that a re-run rewrites before anything reads it: pack_pin, out_pin, c_out, mid, g_out, the router
    buffers, the token id, and the K/V rows at `pos` (each layer graph writes its own row before its attention reads it;
    rings write pos % R). Slot-table updates, captures, PEND flips and GPU admissions the attempt made are real device state
    and stay; the host counters the attempt advanced (GEN, C['cpu'], C['hit'], COUNTS) are put back (server._bell_recover).
  * Exactness: the re-run is the classic loop on the same (tok, pos), the same K/V, the same weights and the residency the
    device holds at that moment. With residency frozen (--refresh-every 0) that is bit-identical to a run that never used
    the doorbell. With adaptive admission the flips of finished admissions (event queries) can land a layer earlier or later
    than in the classic order, so the hit/miss split, and with it the bf16 summation split, can differ from a classic run:
    every token is still an exact computation for the residency it saw (the same is true run-to-run for the classic loop).
  * Cost: <= 36 classic layers (~45 ms) after a >= 800 ms stall: irrelevant. After --doorbell-max-timeouts recoveries in a
    process the early launch is off for good (0 = never). A host-side exception between arm() and ring() still rings
    (try/finally) so the graph never waits for a host that is gone. A timeout INSIDE the classic-order retry (a spin that
    does not exit although FLAG was set before its launch) is a hardware fault, not a stall: it propagates.

MODE SWITCH (choose()).  A page-fault storm is the regime where a CPU kernel stalls for hundreds of ms (first answers after a
cold start: ~1,200 process page faults per token, MEASURED, against ~130 on a first turn and ~20 later), and it is exactly what
the spin's timeout would measure. In the classic order that stall is harmless (the GPU idles); with the doorbell it costs a
recovery. So the early launch is used only in steady state: the first `warm_tokens` decode tokens of every request run
classic, and so does any token after one whose fault count (delta of the process PageFaultCount between two token starts)
exceeded `faults_max`. A classic-mode token in a --doorbell 1 build still runs the spin graphs, with arm()+ring() before each
launch: the spin exits at once (7 us per node, MEASURED in the record).

This module is pure Python (numpy view of a pinned torch tensor) so the protocol can be unit-tested on the CPU
(tests/test_doorbell_mock.py).
"""

NWORDS = 8
FLAG, WANT, TIMEOUT_US, FAILS, POLLS, SPIN_US = range(6)
_M32 = 0xFFFFFFFF
MAX_TIMEOUT_MS = 1800            # the spin must give up well below WDDM's ~2 s TDR
DEFAULT_TIMEOUT_MS = 800
MAX_POLL_CAP = 1 << 20           # ~1 s at the MEASURED 1.1 polls/us (step8_05_bellnf: 44,114 polls in 39.5 ms per token)
DEFAULT_FAULTS_MAX = 300         # first answers ~1,200 faults/token, first turns ~130, later ~20 (MEASURED)
DEFAULT_WARM_TOKENS = 32
DEFAULT_MAX_TIMEOUTS = 3
INJECT_LAYER = 17                # --doorbell-inject withholds the ring of the hand-off after this layer

REQ_KEYS = ("tokens", "classic", "classic_warm", "classic_faults", "classic_off", "classic_timeout", "timeouts", "polls",
            "spin_us")


class DoorbellTimeout(RuntimeError):
    """A graph's spin node gave up waiting for the host (see the module docstring)."""


def s32(x):
    """The int32 with the same bit pattern as x mod 2**32 (numpy refuses to store an out-of-range Python int)."""
    x &= _M32
    return x - (1 << 32) if x >= (1 << 31) else x


def passed(flag, want):
    """GPU-side test of the spin (fused_core.k_bell): true when the flag has reached `want`, in wrapping int32
    arithmetic. Kept here so the mock GPU and the unit tests use the exact predicate the kernel implements."""
    return s32(flag - want) >= 0


class Doorbell:
    def __init__(self, pin, timeout_ms=DEFAULT_TIMEOUT_MS, faults_max=DEFAULT_FAULTS_MAX, warm_tokens=DEFAULT_WARM_TOKENS,
                 max_timeouts=DEFAULT_MAX_TIMEOUTS, inject=0):
        """pin: an int32 tensor with >= NWORDS elements in pinned host memory (its numpy() view is the word array
        the kernel is given a pointer to). The Doorbell owns all host-side writes to it."""
        assert pin.dtype.is_floating_point is False and pin.numel() >= NWORDS
        self.pin = pin
        self.b = pin.numpy()
        assert self.b.dtype.itemsize == 4
        self.seq = 0                 # the last armed sequence number, unwrapped mod 2**32
        self.armed = False
        self.early = True            # launch each spin graph before the CPU kernel that feeds it (off after max_timeouts)
        self.fails_seen = 0
        self.timeouts = 0            # spins that gave up (process lifetime)
        self.recoveries = 0          # tokens re-run in the classic order because of that (process lifetime)
        self.stamp = None            # (layer, E, launch call s, arm->ring s, CPU kernel s) of the latest early hand-off
        self.trace = None            # optional callback(event, seq): the mock test records the host's protocol steps
        assert 0 < timeout_ms <= MAX_TIMEOUT_MS, "the spin must give up well below WDDM's ~2 s TDR"
        assert faults_max >= 0 and warm_tokens >= 0 and max_timeouts >= 0 and inject >= 0
        self.faults_max, self.warm_tokens, self.max_timeouts = faults_max, warm_tokens, max_timeouts
        self.inject = inject         # TEST ONLY: every inject-th early token withholds one ring (0 = never)
        self.early_seen = 0
        self.b[:] = 0
        self.b[TIMEOUT_US] = int(timeout_ms) * 1000
        self.begin_request()

    # ---- per-request mode switch and statistics
    def begin_request(self):
        """A new request starts decoding: the warm-up window restarts, the fault baseline is unknown, the counters zero."""
        self.n_req = 0
        self.pf_last = None
        self.faults_prev = None
        self.req = dict.fromkeys(REQ_KEYS, 0)

    def choose(self, pf_now):
        """The per-token mode switch. pf_now = the process's cumulative PageFaultCount at the START of this token. Returns
        (early, reason): early=True -> launch each graph before its CPU kernel; otherwise the classic order, reason says why
        ('off' = disabled after max_timeouts, 'warm' = first warm_tokens tokens of the request or no fault baseline yet,
        'faults' = the previous token had more than faults_max faults)."""
        n = self.n_req
        self.n_req += 1
        prev, self.pf_last = self.pf_last, pf_now
        self.faults_prev = None if prev is None else (pf_now - prev) & _M32
        if not self.early:
            why = "off"
        elif n < self.warm_tokens or self.faults_prev is None:
            why = "warm"
        elif self.faults_prev > self.faults_max:
            why = "faults"
        else:
            return True, None
        self.req["classic"] += 1
        self.req["classic_" + why] += 1
        return False, why

    def stats(self, n):
        """The per-request keys the server adds to its stats (n = decode tokens of the request)."""
        r, n = self.req, max(n, 1)
        return {"doorbell": 1, "doorbell_tokens": r["tokens"], "doorbell_classic_tokens": r["classic"],
                "doorbell_timeouts": r["timeouts"],
                "doorbell_classic_warm": r["classic_warm"], "doorbell_classic_faults": r["classic_faults"],
                "doorbell_classic_timeout": r["classic_timeout"], "doorbell_classic_off": r["classic_off"],
                "doorbell_polls_per_tok": r["polls"] / n, "doorbell_spin_ms_per_tok": r["spin_us"] / n / 1e3,
                "doorbell_disabled": not self.early}

    def drop_layer(self):
        """TEST ONLY (--doorbell-inject N): the layer whose ring this early token withholds (-1 = none). Every N-th early
        token: the spin of the next graph then has to time out and the token has to be recovered."""
        if not self.inject:
            return -1
        self.early_seen += 1
        return INJECT_LAYER if self.early_seen % self.inject == 0 else -1

    # ---- the protocol
    def open(self):
        """FLAG := WANT: any spin that starts now exits at once. For eager warm-up runs of a graph body and
        before graph capture (nothing is in flight then), and after a timeout (nothing is in flight either)."""
        assert not self.armed
        self.b[FLAG] = self.b[WANT]

    def arm(self):
        """Publish the sequence number of the next hand-off. Call BEFORE the launch of the graph that waits for it,
        and only after the previous spin has finished (the router event of the previous graph was observed)."""
        assert not self.armed, "arm() twice without ring()"
        self.armed = True
        self.seq = (self.seq + 1) & _M32
        self.b[WANT] = s32(self.seq)
        if self.trace is not None:
            self.trace("arm", self.seq)
        return self.seq

    def ring(self):
        """Release the spin: the CPU kernel's output (and the sfence after it) are complete."""
        assert self.armed, "ring() without arm()"
        self.armed = False
        self.b[FLAG] = s32(self.seq)
        if self.trace is not None:
            self.trace("ring", self.seq)

    def drop(self):
        """TEST ONLY: the ring of this hand-off is lost (armed -> not armed, FLAG untouched): the spin has to give up."""
        assert self.armed, "drop() without arm()"
        self.armed = False
        if self.trace is not None:
            self.trace("ring-lost", self.seq)

    def describe(self):
        """One line for the log when a spin gave up. spin_us is the GPU's own clock from the spin's START (a late launch
        can only shorten it); the stamp is the host's view of the same hand-off: if arm->ring is far below the timeout the
        ring existed long before the GPU gave up (GPU starved / not seeing the flag); if it is about the timeout the host
        was stalled, in the CPU kernel (kernel ~ arm->ring) or elsewhere in the window (launch call, flush)."""
        spin_ms = int(self.b[SPIN_US]) / 1e3
        if self.stamp is None:
            return f"GPU spun {spin_ms:.0f} ms of {int(self.b[TIMEOUT_US]) / 1e3:.0f} ms; no host stamp"
        L, E, tl, a2r, kd = self.stamp
        return (f"GPU spun {spin_ms:.0f} ms of {int(self.b[TIMEOUT_US]) / 1e3:.0f} ms; host, hand-off after layer {L}: E={E}, "
                f"launch call+flush {tl * 1e3:.1f} ms, CPU kernel {kd * 1e3:.1f} ms, arm->ring {a2r * 1e3:.1f} ms")

    def read(self):
        """Call after the router event of a graph that starts with a spin node: (gave_up, polls, spin_us) of the spin that
        just finished. gave_up is true once per timeout. Also accumulates the request's spin telemetry."""
        f, n, s = self.b[FAILS:SPIN_US + 1].tolist()
        self.req["polls"] += n
        self.req["spin_us"] += s
        if f != self.fails_seen:
            self.fails_seen = f
            self.timeouts += 1
            self.req["timeouts"] += 1
            return True, n, s
        return False, n, s

    def recovered(self):
        """A token was re-run in the classic order after a timeout. Returns True when that was the last one allowed."""
        self.recoveries += 1
        self.req["classic"] += 1
        self.req["classic_timeout"] += 1
        if self.max_timeouts and self.recoveries >= self.max_timeouts:
            self.early = False
        return not self.early
