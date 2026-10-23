"""Persistent pool of env worker processes; each worker owns many AlfGame instances keyed by caller-chosen ids.

Create the pool BEFORE initialising vLLM/CUDA (workers are forked).
All calls are batched: one message per worker per call, executed in parallel across workers.
"""
import multiprocessing as mp
import traceback


def _worker(conn):
    import alfenv
    games = {}
    idle = []
    while True:
        cmd, payload = conn.recv()
        if cmd == "quit":
            break
        out = {}
        try:
            if cmd == "open":
                for key, game_file, prefix, expected in payload:
                    if idle:
                        g = idle.pop()
                        g.load(game_file)
                    else:
                        g = alfenv.AlfGame(game_file)
                    obs, adm, won, div = g.refeed(prefix, expected)
                    games[key] = g
                    out[key] = (obs, adm, won, div)
            elif cmd == "step":
                for key, action in payload:
                    out[key] = games[key].step(action)
            elif cmd == "open_step":
                # async engine: open + re-feed prefix, then execute the continuation's first action
                for key, game_file, prefix, expected, first_action in payload:
                    if idle:
                        g = idle.pop()
                        g.load(game_file)
                    else:
                        g = alfenv.AlfGame(game_file)
                    obs, adm, won, div = g.refeed(prefix, expected)
                    games[key] = g
                    out[key] = ((obs, adm, won, div), g.step(first_action))
            elif cmd == "close":
                for key in payload:
                    g = games.pop(key, None)
                    if g is not None:
                        idle.append(g)
            conn.send(("ok", out))
        except Exception:
            conn.send(("err", traceback.format_exc()))


class EnvPool:
    def __init__(self, n_workers=20):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.owner = {}
        self._rr = 0

    def _scatter(self, cmd, items, key_of):
        per = [[] for _ in self.conns]
        for it in items:
            per[self.owner[key_of(it)]].append(it)
        active = [i for i, b in enumerate(per) if b]
        for i in active:
            self.conns[i].send((cmd, per[i]))
        out = {}
        for i in active:
            status, res = self.conns[i].recv()
            if status != "ok":
                raise RuntimeError(f"env worker {i} failed:\n{res}")
            out.update(res)
        return out

    def open(self, jobs):
        """jobs: [(key, game_file, prefix_actions, expected_obs_or_None)] -> {key: (obs, admissible, won, n_div)}"""
        counts = [0] * len(self.conns)
        for worker in self.owner.values():
            counts[worker] += 1
        for j in jobs:
            if j[0] in self.owner:
                raise ValueError(f"Environment key already open: {j[0]}")
            worker = min(range(len(counts)), key=counts.__getitem__)
            self.owner[j[0]] = worker
            counts[worker] += 1
        return self._scatter("open", jobs, lambda j: j[0])

    def step(self, actions):
        """actions: {key: action} -> {key: (obs, admissible, won, done)}"""
        return self._scatter("step", list(actions.items()), lambda kv: kv[0])

    def close(self, keys):
        self._scatter("close", list(keys), lambda k: k)
        for k in keys:
            self.owner.pop(k, None)

    # ------------------------------------------------------------ non-blocking API (async replay engine)
    # One message in flight per worker; further requests are buffered per worker and sent as one batch when the
    # worker becomes free. Do not mix with the blocking calls above while async requests are outstanding.

    def submit(self, cmd, items, key_of):
        if not hasattr(self, "_pend"):
            self._pend = [dict() for _ in self.conns]
            self._busy = [None] * len(self.conns)
        for it in items:
            k = key_of(it)
            if cmd in ("open_step", "open"):
                counts = [0] * len(self.conns)
                for w in self.owner.values():
                    counts[w] += 1
                if k in self.owner:
                    raise ValueError(f"Environment key already open: {k}")
                self.owner[k] = min(range(len(counts)), key=counts.__getitem__)
            self._pend[self.owner[k]].setdefault(cmd, []).append(it)
        # close: owner entries are removed when the close is acknowledged (poll)

    def _dispatch(self):
        for i, c in enumerate(self.conns):
            if self._busy[i] is None and self._pend[i]:
                # priority: close (frees slots) > step (finishes work) > open_step (starts work)
                for cmd in ("close", "step", "open_step", "open"):
                    if cmd in self._pend[i]:
                        items = self._pend[i].pop(cmd)
                        c.send((cmd, items))
                        self._busy[i] = (cmd, items)
                        break

    def poll(self, timeout=0.0):
        """Dispatch buffered requests and collect finished ones. Returns list of (cmd, {key: result})."""
        if not hasattr(self, "_pend"):
            return []
        self._dispatch()
        import multiprocessing.connection as mpc
        busy = [c for i, c in enumerate(self.conns) if self._busy[i] is not None]
        ready = mpc.wait(busy, timeout=timeout) if busy else []
        out = []
        for c in ready:
            i = self.conns.index(c)
            status, res = c.recv()
            cmd, items = self._busy[i]
            self._busy[i] = None
            if status != "ok":
                raise RuntimeError(f"env worker {i} failed:\n{res}")
            if cmd == "close":
                for k in items:
                    self.owner.pop(k, None)
            out.append((cmd, res))
        self._dispatch()
        return out

    def idle_async(self):
        return not hasattr(self, "_pend") or (all(b is None for b in self._busy) and not any(self._pend))

    def shutdown(self):
        for c in self.conns:
            c.send(("quit", None))
        for p in self.procs:
            p.join(timeout=5)
