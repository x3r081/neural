"""First-GPU-minute test of the doorbell (fused_core.k_bell + doorbell.Doorbell). Needs the GPU; NOT the server.

    python tools/doorbell_selftest.py                 # stages 0-4, stops at the first failure
    python tools/doorbell_selftest.py --stage 5       # the layer-loop microbenchmark (classic vs doorbell order)

Every stage that can spin has a host-side watchdog that rings the bell after 250 ms, so a spin can end by (a) the
flag, (b) the kernel's own timeout (--timeout-ms, default 50, hard cap 200) or (c) the poll cap, and never runs into
WDDM's ~2 s TDR even if a later stage's assumption is wrong. What each stage proves:

  0  the Triton kernel compiles (while + tl.condition, volatile load, %globaltimer) and exits at once when FLAG >= WANT
  1  the timeout path: FLAG < WANT and nobody rings -> the kernel gives up after ~timeout, bumps FAILS, reports
     spin_us; and the POLL_CAP bound works independently of the timer (a timeout of 1 s with a cap of 4096 polls)
  2  release from the host while the GPU spins: latency ring -> stream.synchronize() (an upper bound of the GPU-side
     detection latency; includes the sync call itself)
  3  the wrapping int32 compare on the GPU: WANT around 2**31 - 1, 2**31, 2**32 - 1, 0: a stale FLAG never releases
  4  CUDA-graph capture of [k_bell, k_fetch, consumer], replayed 2000x with FRESH random data written into the pinned
     source AFTER the launch and BEFORE the ring: the consumer must see exactly that data every time (memory
     visibility / stale-line check for the flag -> next-kernel read that the server relies on)
  5  microbenchmark of the reorder itself: per 36-layer "token", classic (busy-wait 'CPU kernel' -> launch) vs doorbell
     (launch -> busy-wait -> ring) with a graph of representative GPU work; reports ms/token for both and the
     difference (the number to expect from --doorbell 1, minus what the real host bookkeeping adds)
"""
import argparse, os, statistics, sys, threading, time
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import doorbell as DB                                           # noqa: E402
import fused_core as F                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", help="0..5 or all (= 0..4)")
ap.add_argument("--timeout-ms", type=int, default=50)
ap.add_argument("--cpu-kernel-us", type=int, default=1000, help="stage 5: the busy-waited 'CPU kernel' per layer")
ap.add_argument("--bookkeeping-us", type=int, default=60, help="stage 5: host work between the router event and the launch")
ap.add_argument("--layers", type=int, default=36)
ap.add_argument("--reps", type=int, default=30)
A = ap.parse_args()
assert 1 <= A.timeout_ms <= 200
dev = torch.device("cuda")
H = 2880
WATCHDOG_S = 0.25


def make(timeout_ms=None):
    return DB.Doorbell(torch.zeros(8, dtype=torch.int32).pin_memory(), timeout_ms=timeout_ms or A.timeout_ms)


class Watchdog:
    """Releases any spin after WATCHDOG_S no matter what (writes FLAG := WANT)."""

    def __init__(self, bell):
        self.b = bell.b
        self.t = threading.Timer(WATCHDOG_S, self.fire)
        self.fired = False
        self.t.start()

    def fire(self):
        self.fired = True
        self.b[DB.FLAG] = self.b[DB.WANT]

    def cancel(self):
        self.t.cancel()


def sync():
    torch.cuda.synchronize()


def warm(cap):
    """Triton compiles one kernel per POLL_CAP: do it BEFORE arming anything (a compile takes seconds; the watchdog must not
    fire during it)."""
    b = make()
    b.open()
    F.bell_wait(b.pin, poll_cap=cap)
    sync()


def settle(bell):
    """Back to the quiet state: nothing armed, FLAG == WANT."""
    bell.armed = False
    bell.b[DB.FLAG] = bell.b[DB.WANT]


def stage0():
    bell = make()
    bell.arm()
    bell.ring()                                                  # FLAG == WANT: a spin must exit at once
    t0 = time.perf_counter()
    F.bell_wait(bell.pin, poll_cap=1 << 16)                      # Triton compiles here
    sync()
    t1 = time.perf_counter()
    bell.arm(); bell.ring()
    F.bell_wait(bell.pin, poll_cap=1 << 16)
    sync()
    t2 = time.perf_counter()
    gave_up, polls, spin_us = bell.read()
    print(f"  stage 0: compiled+ran in {t1 - t0:.2f} s, second launch {1e3 * (t2 - t1):.2f} ms; gave_up={gave_up} polls={polls} spin_us={spin_us}")
    assert not gave_up and polls < 16 and spin_us < 5000, (gave_up, polls, spin_us)


def stage1():
    warm(1 << 23)
    warm(1 << 12)
    bell = make(30)                                              # 30 ms
    bell.arm()                                                   # armed, never rung
    wd = Watchdog(bell)
    t0 = time.perf_counter()
    F.bell_wait(bell.pin, poll_cap=1 << 23)
    sync()
    wall = time.perf_counter() - t0
    wd.cancel()
    gave_up, polls, spin_us = bell.read()
    print(f"  stage 1a: timeout 30 ms -> gave_up={gave_up} polls={polls} spin_us={spin_us} wall={wall * 1e3:.1f} ms (watchdog fired: {wd.fired})")
    assert not wd.fired and gave_up and 25_000 <= spin_us <= 150_000, "the timer bound did not fire in time"
    settle(bell)
    bell.b[DB.TIMEOUT_US] = 1_000_000                            # 1 s timer, but a cap of 4096 polls: the cap must win
    bell.arm()
    wd = Watchdog(bell)
    t0 = time.perf_counter()
    F.bell_wait(bell.pin, poll_cap=1 << 12)
    sync()
    wall = time.perf_counter() - t0
    wd.cancel()
    gave_up, polls, spin_us = bell.read()
    print(f"  stage 1b: cap 4096 polls -> gave_up={gave_up} polls={polls} spin_us={spin_us} wall={wall * 1e3:.1f} ms (watchdog fired: {wd.fired})")
    assert not wd.fired and gave_up and polls == 4096, "the poll cap did not bound the spin"
    settle(bell)


def stage2(iters=200):
    warm(1 << 23)
    bell = make(200)
    lat, polls_l = [], []
    for _ in range(iters):
        bell.arm()
        F.bell_wait(bell.pin, poll_cap=1 << 23)
        time.sleep(0.002)                                        # the GPU is in the spin by now
        t = time.perf_counter()
        bell.ring()
        torch.cuda.current_stream().synchronize()
        lat.append(time.perf_counter() - t)
        gave_up, polls, spin_us = bell.read()
        assert not gave_up and polls > 0, (gave_up, polls)
        polls_l.append(polls)
    lat_us = sorted(x * 1e6 for x in lat)
    print(f"  stage 2: ring -> synchronize() returns: median {lat_us[len(lat_us) // 2]:.0f} us, p90 {lat_us[int(0.9 * len(lat_us))]:.0f} us, "
          f"max {lat_us[-1]:.0f} us; polls per 2 ms spin median {statistics.median(polls_l):.0f}")


def stage3():
    warm(1 << 23)
    bell = make(200)
    for want in (0x7FFFFFFE, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0, 1):
        prev = (want - 1) & 0xFFFFFFFF
        bell.seq = prev
        bell.b[DB.FLAG] = bell.b[DB.WANT] = DB.s32(prev)         # the state after the previous hand-off
        bell.arm()                                               # WANT = want; FLAG (prev) is stale
        F.bell_wait(bell.pin, poll_cap=1 << 23)
        time.sleep(0.004)
        still_spinning = not torch.cuda.current_stream().query()
        bell.ring()
        sync()
        gave_up, polls, spin_us = bell.read()
        print(f"  stage 3: want={want:#010x}: stale flag held the spin for 4 ms: {still_spinning}; released by ring; gave_up={gave_up} polls={polls}")
        assert still_spinning and not gave_up, "wrapping compare wrong on the GPU"


def stage4(iters=2000):
    bell = make(200)
    src = torch.zeros(1, H, dtype=torch.float32).pin_memory()
    srcnp = src.numpy()
    c_out = torch.zeros(1, H, dtype=torch.float32, device=dev)
    seen = torch.zeros(1, H, dtype=torch.float32, device=dev)

    def body():
        F.bell_wait(bell.pin, poll_cap=1 << 23)
        F.fetch(src, c_out)                                      # the server's k_fetch, unchanged
        seen.copy_(c_out)                                        # a consumer kernel behind it
    bell.open()
    body()
    sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    rng = np.random.default_rng(0)
    early_gpu = 0
    stream = torch.cuda.current_stream()
    for it in range(iters):
        bell.arm()
        g.replay()
        stream.query()                                           # the doorbell flush
        t_end = time.perf_counter() + rng.random() * 300e-6      # the 'CPU kernel'
        while time.perf_counter() < t_end:
            pass
        srcnp[:] = rng.standard_normal((1, H)).astype(np.float32)
        bell.ring()
        sync()
        got = seen.cpu().numpy()
        gave_up, polls, spin_us = bell.read()
        early_gpu += polls == 0
        assert not gave_up, f"iteration {it}: the spin gave up"
        assert np.array_equal(got, srcnp), f"iteration {it}: the consumer saw stale/partial data ({int((got != srcnp).sum())} of {H} differ)"
    print(f"  stage 4: {iters} replays with fresh data between launch and ring: all exact; spins that saw the flag on the first poll: {early_gpu}")


def stage5():
    L, reps = A.layers, A.reps
    bell = make(1000)
    src = torch.zeros(1, H, dtype=torch.float32).pin_memory()
    srcnp = src.numpy()
    c_out = torch.zeros(1, H, dtype=torch.float32, device=dev)
    x = torch.randn(1, H, dtype=torch.bfloat16, device=dev)
    w = torch.randn(H, H, dtype=torch.bfloat16, device=dev) * 0.02
    ev = torch.cuda.Event(external=True)

    def work(with_bell):
        if with_bell:
            F.bell_wait(bell.pin, poll_cap=1 << 23)
        F.fetch(src, c_out)
        y = x
        for _ in range(8):                                       # ~8 x (16 MiB read): a few hundred us of GPU work
            y = (y @ w) * 0.5
        c_out.add_(y.float().sum())
        ev.record()
        y2 = y
        for _ in range(4):                                       # 'b_body' after the event
            y2 = (y2 @ w) * 0.5
        x.copy_(y2)
    graphs = {}
    for name, wb in (("plain", False), ("bell", True)):
        bell.open()
        work(wb)
        sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            work(wb)
        graphs[name] = g
    stream = torch.cuda.current_stream()

    def busy(us):
        t_end = time.perf_counter() + us * 1e-6
        while time.perf_counter() < t_end:
            pass

    def token_classic():
        t0 = time.perf_counter()
        graphs["plain"].replay()
        for i in range(L):
            ev.synchronize()
            busy(A.bookkeeping_us)                               # parse + bookkeeping
            busy(A.cpu_kernel_us)                                # the CPU expert kernel
            srcnp[:] = i
            graphs["plain"].replay()                             # launch AFTER the kernel (today)
        sync()
        return time.perf_counter() - t0

    def token_bell():
        t0 = time.perf_counter()
        graphs["plain"].replay()
        for i in range(L):
            ev.synchronize()
            busy(A.bookkeeping_us)
            bell.arm()
            graphs["bell"].replay()                              # launch BEFORE the kernel
            stream.query()
            busy(A.cpu_kernel_us)
            srcnp[:] = i
            bell.ring()
        sync()
        gave_up, polls, spin_us = bell.read()
        assert not gave_up
        return time.perf_counter() - t0

    for f in (token_classic, token_bell, token_classic, token_bell):
        f()
    res = {"classic": [], "doorbell": []}
    for _ in range(reps):
        res["classic"].append(token_classic())
        res["doorbell"].append(token_bell())
    mc, mb = (statistics.median(res[k]) * 1e3 for k in ("classic", "doorbell"))
    print(f"  stage 5: {L} layers, cpu kernel {A.cpu_kernel_us} us, bookkeeping {A.bookkeeping_us} us: classic {mc:.2f} ms/token, "
          f"doorbell {mb:.2f} ms/token, difference {mc - mb:+.2f} ms/token ({100 * (mc - mb) / mc:+.1f}%); "
          f"per layer {1e3 * (mc - mb) / L:+.1f} us")


STAGES = {"0": stage0, "1": stage1, "2": stage2, "3": stage3, "4": stage4, "5": stage5}

if __name__ == "__main__":
    which = ["0", "1", "2", "3", "4"] if A.stage == "all" else [A.stage]
    print(f"doorbell selftest on {torch.cuda.get_device_name(0)}, torch {torch.__version__}, timeout {A.timeout_ms} ms", flush=True)
    for s in which:
        STAGES[s]()
    print("done", flush=True)
