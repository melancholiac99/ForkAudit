"""Mocked publication/order checks; no network mutation and no actual shutdown."""
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import publish_completed as P
import repair_guardian as G


def main():
    checks = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); file = root/'report.json'; file.write_text('{}\n')
        calls = []; staged = [False]

        def fake_git(where, *args, **kw):
            calls.append(args)
            if args[:3] == ('remote', 'get-url', '--push'): return P.REMOTE
            if args[:2] == ('branch', '--show-current'): return 'main'
            if args == ('diff', '--cached', '--name-only'): return 'report.json' if staged[0] else ''
            if args == ('diff', '--name-only'): return ''
            if args[0] == 'add': staged[0] = True
            if args[0] == 'commit': staged[0] = False
            if args == ('rev-parse', 'HEAD'): return 'a'*40
            if args[0] == 'ls-remote': return 'a'*40+'\trefs/heads/server/test'
            return ''

        with patch.object(P, 'git', fake_git):
            out = P.publish(root, [file], 'server/test', 'main', 'test')
        assert out['verified']
        assert ('add', '-f', '--', 'report.json') in calls
        assert ('push', 'origin', 'HEAD:refs/heads/server/test') in calls
        assert not any(c[0] in ('checkout', 'reset', 'clean') or '--force' in c for c in calls)
        checks.append('exact file allowlist, current branch preserved, no force push, remote SHA checked')

        def dirty_git(where, *args, **kw):
            if args == ('diff', '--name-only'): return 'unrelated-user-file.txt'
            return fake_git(where, *args, **kw)

        with patch.object(P, 'git', dirty_git):
            try: P.publish(root, [file], 'server/test', 'main', 'test')
            except AssertionError: pass
            else: raise AssertionError('Unrelated dirty file accepted')
        checks.append('unrelated tracked changes rejected without overwrite')
        cfg = dict(platform_shutdown_sha256=G.COMPLETE.digest('/usr/bin/shutdown'), publisher_sha256=G.COMPLETE.digest(P.__file__))
        for fails in (False, True):
            order = []
            def publish():
                order.append('publish')
                if fails: raise RuntimeError('simulated upload failure')
                return dict(verified=True)
            with patch.object(P, 'success', publish), patch.object(G, 'status', lambda *a, **kw: None), \
                 patch.object(G, 'save', lambda *a, **kw: order.append('save_failure')), \
                 patch.object(G.os, 'sync', lambda: order.append('sync')), \
                 patch.object(G.subprocess, 'run', lambda *a, **kw: order.append('shutdown')), \
                 patch.object(G.traceback, 'print_exc', lambda: None):
                G.shutdown(cfg, 'both_experiments_verified')
            assert order[0] == 'publish' and order[-2:] == ['sync', 'shutdown']
            if fails: assert order.index('save_failure') < order.index('shutdown')
        checks.append('publication precedes shutdown; failed upload saved locally before shutdown')
    assert '[REDACTED]' in P.redact('Authorization: Bearer abcdef')
    result = dict(passed=True, checks=checks, network_mutations=False, real_shutdown=False)
    P.save(P.H/'run/publication_selftest.json', result)
    print(result)


if __name__ == '__main__': main()
