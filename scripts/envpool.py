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

    def shutdown(self):
        for c in self.conns:
            c.send(("quit", None))
        for p in self.procs:
            p.join(timeout=5)
