"""Prompt-cache bookkeeping for a rolling (ring) K/V cache on the sliding-window layers.

With `FusedCore(ring=R)` each of the 18 sliding layers keeps only the last R positions of K/V
(slot = position % R). Decode and prompt processing are bit-identical to the linear cache
(fused_core.py), but a prompt that resumes from an EARLIER position c needs the window before
it (positions c-W+1 .. c-1) to be present in the ring. A slot holds the LAST position written to
it, so position q is present iff no later write hit q + kR. Two facts follow:

  * after a resume at c that ends at P < n (the old end), slots of positions P..n-1 still hold
    that old data, which aliases positions P-R..n-1-R: the ring's usable range is bounded by the
    HIGH-WATER MARK of everything ever written since the last restore, not by len(CACHE);
  * a prompt pass that fails half-way (CUDA OOM) has already overwritten slots below c:
    the ring is DIRTY until a checkpoint is restored or the prompt is re-processed from 0.

This module keeps CHECKPOINTS: host snapshots of the 18 rings taken at the end of every
prompt pass (9 MiB each at R=256), each with the high-water mark it was taken at. Once the first
generated token has been decoded the snapshot is taken again one position later, replacing the
first: a follow-up re-renders the answer, so it shares that token (the one that opens the assistant
turn) but rarely more, and its K/V is the decode path's; the linear cache keeps it, and a
checkpoint at the prompt end would recompute it through prefill (equal only up to bf16). On a rollback
to c the live rings are used when their window is intact; otherwise the checkpoint that can serve c
is restored, preferring one taken at or after c (no re-processing at all: regenerate-last-turn),
then the newest one at or before c (re-process from there), else the prompt is re-processed from
0. All of it is exact: identical tokens give identical K/V, and positions before the resume point
are untouched.

Pure Python (snapshot payloads are opaque objects supplied by the caller): tests/test_kv_ring.py.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def ring_reuse_ok(n_written: int, c: int, ring: int, window: int = 128) -> bool:
    """fused_core.ring_reuse_ok with n_written = HIGH-WATER MARK of positions written, plus:
    resuming from 0 never needs the ring."""
    if c <= 0:
        return True
    if c > n_written:
        return False
    if n_written <= ring:
        return True
    return n_written - c <= ring - window + 1


class RingCheckpoints:
    """Checkpoints (position, snapshot, high-water mark), ordered by position."""

    def __init__(self, ring: int, window: int = 128, max_keep: int = 8,
                 restore: Optional[Callable[[Any], None]] = None) -> None:
        assert ring >= window > 0
        self.ring, self.window, self.max_keep = int(ring), int(window), max(1, int(max_keep))
        self._restore = restore
        self._cps: list[tuple[int, Any, int]] = []
        self.hw = 0                 # highest position + 1 ever written into the live rings since the last restore
        self.dirty = False          # a prompt pass started and has not finished: live rings are not trustworthy
        self.stats = {"taken": 0, "restored_at": 0, "restored_before": 0, "full_reprefill": 0, "live_ok": 0}

    # ------------------------------------------------------------------ bookkeeping
    def positions(self) -> list[int]:
        return [p for p, _, _ in self._cps]

    def note_written(self, n: int) -> None:
        """Positions 0..n-1 have been written (prefill end or decode progress)."""
        self.hw = max(self.hw, int(n))

    def begin_write(self, c: int) -> None:
        """A prompt pass is about to overwrite positions >= c."""
        self.dirty = True
        self.invalidate_from(c)

    def take(self, pos: int, snap: Any, replaces: Optional[int] = None) -> None:
        """Checkpoint right after positions up to pos-1 were written (clears dirty). `replaces` is the
        position of this request's earlier checkpoint that this one supersedes (it would take a slot)."""
        self.note_written(pos)
        self.dirty = False
        self._cps = [(p, s, h) for p, s, h in self._cps if p < pos and p != replaces]
        self._cps.append((int(pos), snap, self.hw))
        while len(self._cps) > self.max_keep:
            del self._cps[1 if len(self._cps) > 2 else 0]   # keep the oldest (deep rollback) and the newest
        self.stats["taken"] += 1

    def invalidate_from(self, c: int) -> None:
        """Positions >= c are about to be rewritten: checkpoints taken beyond c no longer describe the cache."""
        self._cps = [(p, s, h) for p, s, h in self._cps if p <= c]

    def export_list(self):
        return (list(self._cps), self.hw, self.dirty)

    def import_list(self, st) -> None:
        if st is None:
            self._cps, self.hw, self.dirty = [], 0, False
        else:
            cps, hw, dirty = st
            self._cps, self.hw, self.dirty = list(cps), int(hw), bool(dirty)

    # ------------------------------------------------------------------ the decision
    def plan(self, c: int, n_written: int):
        """Where must prompt processing resume when the caller wants position c (the longest common
        prefix with the cached tokens), given that positions 0..n_written-1 are in the token cache?
        Returns (c', checkpoint | None) without touching the rings:
          (c, None)       the live rings hold c's window;
          (c, cp)         checkpoint cp (taken at or after c) holds it: restore, keep c;
          (p, cp)         checkpoint cp taken at p < c: restore, re-process from p;
          (0, None)       nothing usable: re-process from 0."""
        c = int(c)
        self.note_written(n_written)
        if not self.dirty and ring_reuse_ok(self.hw, c, self.ring, self.window):
            return c, None
        after = [cp for cp in self._cps if cp[0] >= c and ring_reuse_ok(cp[2], c, self.ring, self.window)]
        if after:
            return c, min(after, key=lambda cp: cp[0])
        before = [cp for cp in self._cps if cp[0] <= c and ring_reuse_ok(cp[2], cp[0], self.ring, self.window)]
        if before:
            cp = max(before, key=lambda cp: cp[0])
            return cp[0], cp
        return 0, None

    def commit(self, c2: int, cp) -> int:
        """Apply a plan(): restore the checkpoint if any, reset the marks, drop what the resume invalidates."""
        if cp is not None:
            if self._restore is not None:
                self._restore(cp[1])
            self.hw = cp[2]
            self.dirty = False
            self.stats["restored_at" if cp[0] > c2 else "restored_before"] += 1
        elif c2 == 0:
            self.stats["full_reprefill"] += 1
            self._cps.clear()
            self.hw = 0
            self.dirty = False
        else:
            self.stats["live_ok"] += 1
        self.invalidate_from(c2)
        return c2

    def rewind(self, c: int, n_written: int) -> int:
        """plan() + commit(). Returns c' <= c to resume from; the caller truncates its token cache to c'
        and then calls begin_write(c') before processing (the K/V of c'..c-1 are recomputed identically)."""
        c2, cp = self.plan(c, n_written)
        return self.commit(c2, cp)
