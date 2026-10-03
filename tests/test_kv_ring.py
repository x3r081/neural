"""Unit tests for kv_ring.RingCheckpoints (pure Python; no model, no torch).

A RingModel tracks, per slot, the LAST position written (exactly what the real rolling cache
holds), so stale aliasing after partial rewinds is modelled faithfully. Every resume position the
policy returns is checked against that model: the window before it must be intact.

    python tests/test_kv_ring.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_ring import RingCheckpoints, ring_reuse_ok

W = 128


class RingModel:
    def __init__(self, R):
        self.R, self.n, self.slots = R, 0, {}

    def write(self, q):
        self.slots[q % self.R] = q

    def write_to(self, n):                  # positions self.n .. n-1
        for q in range(self.n, n):
            self.write(q)
        self.n = n

    def snap(self):
        return (self.n, dict(self.slots))

    def restore(self, s):
        self.n, self.slots = s[0], dict(s[1])

    def holds(self, q):
        return self.slots.get(q % self.R) == q

    def window_ok(self, c):
        return all(self.holds(q) for q in range(max(0, c - W + 1), c))


class Sim:
    """Drives RingCheckpoints the way server.generate()/canon_job() do."""

    def __init__(self, R, keep=8):
        self.m = RingModel(R)
        self.cps = RingCheckpoints(R, W, keep, restore=self.m.restore)
        self.cache = 0                      # len(CACHE)
        self.writing = False                # a resume() opened a prompt pass that prefill()/finish() completes
        self.prompt_end = None              # end of the last prompt pass, until the first decode step follows it

    def prefill(self, n_total, fail=False):
        """prompt pass from the current cache end to n_total (checkpoint at the end unless it fails).
        A plain continuation (no resume() before it) must never rewind."""
        c = self.cache
        if not self.writing:
            c2, cp = self.cps.plan(c, self.cache)
            assert c2 == c and cp is None, "continuation must never rewind"
            self.cps.begin_write(c)
        self.writing = False
        if fail:
            self.m.write_to(c + max(1, (n_total - c) // 2))   # half the block landed
            self.m.n = c                                       # the caller keeps CACHE at c
            return
        self.m.write_to(n_total)
        self.cache = n_total
        self.cps.take(n_total, self.m.snap())
        self.prompt_end = n_total

    def decode(self, n_total):
        """decode steps up to n_total. Like server.generate(), the first step after a prompt pass takes the
        checkpoint a follow-up resumes from one position later, superseding the prompt-end one."""
        p, self.prompt_end = self.prompt_end, None
        if p is not None and n_total > p:
            self.m.write_to(p + 1)
            self.cache = p + 1
            self.cps.take(p + 1, self.m.snap(), replaces=p)
        self.m.write_to(n_total)
        self.cache = n_total
        self.cps.note_written(n_total)

    def resume(self, c):
        """new prompt diverging at c: returns the position processing really resumes from."""
        c2 = self.cps.rewind(c, self.cache)
        assert 0 <= c2 <= c, (c, c2)
        assert c2 == 0 or self.m.window_ok(c2), (c, c2, self.m.n, self.cps.hw)
        self.cache = c2
        self.m.n = c2
        self.cps.begin_write(c2)
        self.writing = True
        return c2

    def finish(self, n_total, fail=False):
        self.prefill(n_total, fail=fail)


def test_predicate():
    assert ring_reuse_ok(5000, 5000, 256) and ring_reuse_ok(5000, 4900, 256)
    assert not ring_reuse_ok(5000, 4800, 256)
    assert ring_reuse_ok(5000, 0, 256) and not ring_reuse_ok(50, 60, 256)


def test_continuation_never_rewinds():
    s = Sim(256)
    s.prefill(13000); s.decode(13512)
    assert s.resume(13512) == 13512


def test_follow_up_reuses_the_first_generated_token():
    # MEASURED (benchmarks/levers_2026-09-29/step1_baseline.json): a follow-up re-renders the answer, so it
    # shares the prompt (13107 tokens) and the first generated token, and the linear cache reports
    # cached_tokens 13108, K/V of position 13107 from decode. The ring path must resume at 13108 too: a
    # checkpoint at the prompt end would re-process that token through prefill (K/V equal only up to bf16).
    s = Sim(256)
    P = 13107
    s.prefill(P); s.decode(P + 511)
    for P_next in (13629, 14148, 14668):                 # the reference conversation's next prompts
        assert s.resume(P + 1) == P + 1, (P, P_next)
        s.finish(P_next); s.decode(P_next + 511)
        P = P_next
    assert s.cps.stats["full_reprefill"] == 0
    assert len(s.cps.positions()) == 4 and s.cps.positions()[-1] == P + 1    # one checkpoint per request


def test_canonical_rerender_uses_first_token_checkpoint():
    # the idle re-render diverges after the first generated token; the slots of the decoded positions
    # 13002.. were reused by the 512 decoded tokens, so the exact option is the checkpoint at 13001 plus
    # the re-processed tokens after it
    s = Sim(256)
    s.prefill(13000); s.decode(13512)
    assert s.resume(13002) == 13001
    assert s.cps.stats["restored_before"] == 1


def test_regenerate_last_token_uses_checkpoint_after_c():
    s = Sim(256)
    s.prefill(5000); s.decode(5300)
    assert s.resume(4999) == 4999


def test_edit_earlier_turn_goes_to_that_turns_checkpoint():
    s = Sim(256)
    s.prefill(5000); s.decode(5400); s.prefill(5900); s.decode(6300); s.prefill(6800); s.decode(7300)
    assert s.resume(6100) == 5901                   # the turn's checkpoint is taken after its first generated token


def test_side_request_full_reprefill_when_no_checkpoint_fits():
    s = Sim(256)
    s.prefill(13000); s.decode(13512)
    assert s.resume(200) == 0


def test_short_rewind_served_live():
    s = Sim(256)
    s.prefill(1000); s.decode(1100)
    assert s.resume(1050) == 1050


def test_stale_edge_after_partial_rewind_is_not_reused():
    # old end 5300; live resume at 5290 ending at 5295; slots of 5295..5299 still hold OLD data,
    # which aliases positions 5039..5043. A later rewind into that edge must not be served live.
    s = Sim(256)
    s.prefill(5000); s.decode(5300)
    assert s.resume(5290) == 5290
    s.finish(5295)
    for c in range(5160, 5175):
        s2 = Sim(256); s2.prefill(5000); s2.decode(5300); s2.resume(5290); s2.finish(5295)
        s2.resume(c)                         # the Sim asserts the window is intact whatever it chose


def test_failed_prefill_marks_ring_dirty():
    s = Sim(256)
    s.prefill(5000); s.decode(5300)
    s.resume(5290)
    s.finish(6000, fail=True)                # half the block landed, cache stays at 5290
    c2 = s.resume(5290)                      # retry: live rings are dirty -> checkpoint or 0, never live
    assert c2 == 5001 and s.cps.stats["live_ok"] == 1   # only the first resume was served live
    s.finish(6000)                           # the retry succeeds and re-arms a checkpoint
    assert s.resume(6000) == 6000


def test_checkpoint_cap_keeps_oldest_and_newest():
    s = Sim(256, keep=4)
    n = 0
    for _ in range(20):
        n += 300; s.prefill(n); n += 200; s.decode(n)
    ps = s.cps.positions()
    assert len(ps) == 4 and ps[0] == 301 and ps[-1] == n - 200 + 1


def test_invalidate_beyond_resume():
    s = Sim(256)
    for p in (1000, 2000, 3000):
        s.prefill(p); s.decode(p + 100)
    s.resume(2500)
    assert s.cps.positions() == [1001, 2001], s.cps.positions()


def test_export_import_roundtrip():
    s = Sim(256)
    s.prefill(1000); s.decode(1200)
    st = s.cps.export_list()
    s.cps.import_list(None)
    assert s.cps.positions() == [] and s.cps.hw == 0
    s.cps.import_list(st)
    assert s.cps.positions() == [1001] and s.cps.hw == 1200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
