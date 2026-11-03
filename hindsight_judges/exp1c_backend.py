"""Chunked-prefill execution layer for experiment 1c, wrapping the frozen P4 E2 scorer.

Why it exists
  Hindsight experiments run the 32B judge under SDPA MATH (EXECUTION_AMENDMENT_20261102.txt).
  MATH materialises a heads x N x N attention-score tensor per layer (FP32 upcast for BF16 inputs in
  current PyTorch). 1c's L / LN contexts are ~11-13k tokens and dual / single replay contexts up to
  ~8k; on an 85 GB H800 already holding ~67 GB of 32B weights a one-pass prefill of that length
  cannot fit (40 heads x 12k^2 x 4 B = 23 GB for one score tensor).

What is unchanged
  The model object is the one built by the frozen exploratory_P4/p4_e2_backend.py Scorer (BF16
  transformer, FP32 lm_head, sdpa); the caller asserts that file's SHA-256. Candidate scoring keeps
  the frozen semantics: the first candidate token is scored from the prefix's last position, the
  remaining tokens from one forward pass of the candidate over a deep copy of the prefix KV cache.

What changes
  The prefix (score) or prefix+candidate (direct) is fed in fixed-size chunks with a growing KV cache.
  Every chunk attends to the cache plus itself through the same cached-attention path the frozen
  scorer already uses for candidate suffixes. Peak attention memory becomes heads x CHUNK x N.
"""
import copy

CHUNK = 1024


class ChunkedScorer:
    def __init__(self, frozen, chunk=CHUNK):
        self.frozen = frozen
        self.t = frozen.t
        self.model = frozen.model
        self.chunk = int(chunk)
        assert self.chunk >= 2
        self.device = next(self.model.parameters()).device

    def _ids(self, ids):
        return self.t.tensor([ids], device=self.device)

    def _feed(self, ids, keep_from):
        """Feed `ids` in chunks. Returns (KV cache, FP32 hidden states of positions >= keep_from)."""
        t, m = self.t, self.model
        cache, kept = None, []
        for start in range(0, len(ids), self.chunk):
            part = ids[start:start + self.chunk]
            out = m.model(input_ids=self._ids(part), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            lo = max(keep_from - start, 0)
            if lo < len(part):
                kept.append(out.last_hidden_state[0, lo:].float())
            del out
        return cache, (t.cat(kept, 0) if kept else None)

    def score(self, job):
        """Same return value as the frozen Scorer.score: per candidate, per-token log-probs."""
        t, m = self.t, self.model
        pre = job['prefix_ids']
        cmds = [c['suffix_ids'] for c in job['candidates']]
        with t.inference_mode():
            cache, last = self._feed(pre, len(pre) - 1)
            lp0 = t.log_softmax(m.lm_head(last[-1]), -1)
            del last
            rows = []
            for cmd in cmds:
                kv = copy.deepcopy(cache)
                h = m.model(input_ids=self._ids(cmd[:-1]), past_key_values=kv, use_cache=False).last_hidden_state
                lp = t.log_softmax(m.lm_head(h[0].float()), -1)
                rows.append([float(lp0[cmd[0]])] +
                            lp[t.arange(len(cmd) - 1, device=self.device), t.tensor(cmd[1:], device=self.device)].tolist())
                del kv, h, lp
            del cache
            return rows

    def direct(self, job, arm):
        """Same return value as the frozen Scorer.direct, computed over prefix+candidate in chunks."""
        t, m = self.t, self.model
        pre = job['prefix_ids']
        cmd = job['candidates'][arm]['suffix_ids']
        ids = pre + cmd
        with t.inference_mode():
            cache, h = self._feed(ids, len(pre) - 1)
            del cache
            lp = t.log_softmax(m.lm_head(h[:len(cmd)]), -1)
            return lp[t.arange(len(cmd), device=self.device), t.tensor(cmd, device=self.device)].tolist()
