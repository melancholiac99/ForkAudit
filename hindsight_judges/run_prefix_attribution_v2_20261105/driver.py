"""One-GPU bounded queue: both verified pilots, then full 386-point runs. No shutdown."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

RUN = Path(__file__).resolve().parent
HERE = RUN.parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import exp2_common as C
import exp4_prefix_attribution as P

OUT = HERE / 'results/prefix_attribution_v2_20261105'
GPU = 'GPU-3c9b342e-d6d7-d958-412a-9bf6e74f1b36'
MODELS = {n: '/root/autodl-tmp/models/Qwen2.5-' + n.upper().replace('B', 'B') + '-Instruct' for n in ('7b', '32b')}
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')


def state(stage, **kw):
    obj = dict(stage=stage, utc=P.utc(), pid=os.getpid(), gpu_uuid=GPU, **kw)
    C.save_json(RUN / 'status.json', obj)
    print(json.dumps(obj), flush=True)


def wake(message):
    thread = os.environ.get('CODEX_THREAD_ID')
    if not thread:
        (RUN / 'notification_unavailable.txt').write_text(message)
        return
    with (RUN / 'wake.log').open('a') as f:
        try:
            subprocess.run(['codex', 'queue', '--thread', thread, '--resume', '--message', message],
                           stdout=f, stderr=subprocess.STDOUT, timeout=45, check=True)
        except Exception:
            traceback.print_exc(file=f)


def main():
    with (RUN / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = C.read_json(RUN / 'execution.json')
        assert plan['implementation'] == P.implementation()
        assert plan['gpu_uuid'] == GPU
        jobs = C.read_json(RUN / 'jobs.json') if (RUN / 'jobs.json').exists() else {}
        try:
            P.prepared(OUT)
            assert C.read_json(OUT / 'historical_cpu_audit.json')['passed']
            for limit in (32, 0):
                # Both model pilots must pass before ANY full-cohort scoring.
                if not limit:
                    for model in MODELS:
                        v = C.read_json(OUT / model / 'first32/verification.json')
                        assert v['passed'] and v['historical_anchors_compatible']
                for model, model_path in MODELS.items():
                    for command in ('score', 'analyze', 'verify'):
                        assert plan['implementation'] == P.implementation(), 'Code changed during immutable queue'
                        key = f'{model}_{"first32" if limit else "full"}_{command}'
                        if key in jobs and jobs[key].get('returncode') == 0:
                            continue
                        if command == 'score':
                            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                              '--format=csv,noheader'], text=True)
                            assert not any(GPU in line for line in active.splitlines()), 'Assigned GPU is occupied'
                        args = [sys.executable, '-u', str(HERE / 'exp4_prefix_attribution.py'), command,
                                '--out', str(OUT), '--name', model, '--model', model_path, '--limit', str(limit)]
                        with (RUN / (key + '.log')).open('a') as log:
                            proc = subprocess.Popen(args, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                            jobs[key] = dict(pid=proc.pid, started=time.time(), command=args, gpu_uuid=GPU)
                            C.save_json(RUN / 'jobs.json', jobs)
                            state('running', task=key, child_pid=proc.pid, jobs=jobs)
                            while proc.poll() is None:
                                time.sleep(20)
                                folder = OUT / model / ('first32' if limit else 'full')
                                progress = C.read_json(folder / 'progress.json') if (folder / 'progress.json').exists() else None
                                state('running', task=key, child_pid=proc.pid, progress=progress)
                            jobs[key].update(returncode=proc.returncode, finished=time.time())
                            C.save_json(RUN / 'jobs.json', jobs)
                            assert proc.returncode == 0, f'{key} exited {proc.returncode}; see {key}.log'
                    if limit:
                        run = C.read_json(OUT / model / 'first32/run.json')
                        C.save_json(RUN / f'{model}_pilot_timing.json', dict(contexts=run['n'],
                            seconds_per_context=run['total_scoring_seconds']/run['n'],
                            extrapolated_full_scoring_hours=run['total_scoring_seconds']/run['n']*6176/3600,
                            caveat='First 32 points only; excludes model loading/gates/analysis and may differ for later contexts.'))
                if limit:
                    state('both_pilots_verified')
                    wake('【Prefix v2 两模型试跑通过】八条件 7B/32B 的 first32 已完成、历史锚点比较和独立核验通过。队列按用户授权继续完整 386 点。请读 ' + str(RUN) + '/status.json 及两份 pilot_timing.json 更新实测耗时；不修改运行配置，不运行训练或关机。')
            state('completed_verified', jobs=jobs)
            wake('【Prefix v2 完整实验完成】八条件 7B/32B 各 386 点已评分、分析和独立核验，请读 ' + str(OUT) + '/{7b,32b}/full/analysis.json、verification.json 与 historical_anchor_comparison.json。按最新协议汇报两条主终点、证据斜率/DROP 比和历史锚点；仅发布到 research/hindsight-judges，不修改 main 或旧结果。本队列没有关机动作。')
        except Exception:
            error = traceback.format_exc()
            (RUN / 'failure.log').write_text(error)
            state('failed', error=error, jobs=jobs)
            wake('【Prefix v2 异常】请尽力修复并续跑。读取 ' + str(RUN) + '/status.json、failure.log 和对应日志。保留八条件协议、原数据、TV<0.005 与冻结精度，禁止混用运行配置或覆盖旧实验。未授权训练或关机。')
            raise


if __name__ == '__main__':
    main()
