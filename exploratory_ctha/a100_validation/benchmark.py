"""Fixed real prompt inputs, fresh A100 generations and independent HF scoring."""
import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from a100_common import DATA, FAST, HERE, MODEL, PARENT, ROOT, PeakMemory, fingerprint, hardware, load, save, sha


def prepare(out):
    if out.exists():
        raise ValueError('Reference prompt directory must be fresh')
    sys.path.insert(0, str(ROOT / 'exploratory_credit_projection'))
    from audit import JoinedParts
    archive_root = ROOT / 'exploratory_E4/results_archive'
    manifest = load(archive_root / 'archive_manifest.json')
    parts = []
    for item in manifest['parts']:
        path = archive_root / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Old reference archive checksum mismatch: {path}')
        parts.append(path)
    rows = []
    episodes = []
    stream = JoinedParts(parts)
    try:
        with tarfile.open(fileobj=stream, mode='r|gz') as archive:
            for member in archive:
                if not (member.isfile() and member.name.startswith('data/rollout/train/base/')
                        and member.name.endswith('_r0.json')):
                    continue
                payload = archive.extractfile(member).read()
                episode = json.loads(payload)
                if not episode['steps']:
                    raise ValueError('Selected reference episode has no real decisions')
                episodes.append(member.name)
                for turn in [1, 5, 10, 20]:
                    step = episode['steps'][min(turn, len(episode['steps'])) - 1]
                    rows.append(dict(prompt=step['prompt'], seed=20261113 + len(rows),
                                     archive_member=member.name, archive_member_sha256=__import__('hashlib').sha256(payload).hexdigest(),
                                     actual_turn=step['turn']))
                if len(episodes) == 4:
                    break
    finally:
        stream.close()
    assert len(rows) == 16
    out.mkdir(parents=True)
    save(out / 'prompts.json', rows, immutable=True)
    save(out / 'provenance.json', dict(
        old_hardware='6000D, reference prompt text only', old_scores_used=False,
        old_rewards_used=False, old_training_samples_used=False,
        new_responses='All candidate responses/logprobs generated freshly on A100',
        training_use='Forbidden: all engineering training episodes must be recollected after config freeze',
        parts=manifest['parts'], selected_episodes=episodes,
        prompt_sha256=sha(out / 'prompts.json')), immutable=True)


def sample(config_path, prompts_path, out):
    import torch
    from vllm import LLM, SamplingParams
    if out.exists():
        raise ValueError('Use an independent, fresh candidate output directory')
    out.mkdir(parents=True)
    config = load(config_path)
    rows = load(prompts_path)
    os.environ['CTHA_A100_CONFIG_JSON'] = json.dumps(config, sort_keys=True)
    metadata = dict(config=config, config_fingerprint=fingerprint(config),
                    input_sha256=sha(prompts_path), gpu=hardware(),
                    visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                    source_sha256={p.name: sha(p) for p in [Path(__file__), HERE / 'worker.py', HERE / 'a100_common.py', HERE / 'telemetry.py']})
    import importlib.metadata
    metadata['packages'] = {p: importlib.metadata.version(p) for p in ['torch', 'vllm', 'triton', 'transformers', 'peft']}
    metadata['model_manifest_sha256'] = sha(ROOT / '.runtime/assets_manifest.json')
    save(out / 'registration.json', metadata, immutable=True)
    options = dict(model=str(MODEL), dtype=config['dtype'], seed=0,
                   max_model_len=6144, gpu_memory_utilization=config['gpu_memory_utilization'],
                   enforce_eager=True, enable_prefix_caching=True, generation_config='vllm',
                   enable_lora=False, logprobs_mode='processed_logprobs',
                   max_num_seqs=config['max_num_seqs'])
    if config['worker'] == 'a100':
        options['worker_cls'] = 'exploratory_ctha.a100_validation.worker.A100Worker'
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    memory_monitor = PeakMemory().start()
    try:
        engine = LLM(**options)
    except Exception as error:
        save(out / 'failure.json', dict(error=f'{type(error).__name__}: {error}',
                                       device_peak_used_mib=memory_monitor.finish()))
        raise
    tok = engine.get_tokenizer()
    requests = [tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    params = [SamplingParams(temperature=.7, top_p=1., top_k=-1, min_p=0., max_tokens=512,
                             logprobs=0, seed=r['seed']) for r in rows]
    load_seconds = time.perf_counter() - start
    try:
        warmup = engine.generate(requests, params, use_tqdm=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        responses = engine.generate(requests, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        saved = []
        repeat = []
        for before, response in zip(warmup, responses):
            c = response.outputs[0]
            repeat.append(list(before.outputs[0].token_ids) == list(c.token_ids))
            saved.append(dict(prompt_ids=list(response.prompt_token_ids), completion_ids=list(c.token_ids),
                              behavior_logprobs=[float(v[t].logprob) for t, v in zip(c.token_ids, c.logprobs)],
                              response=c.text))
        assert saved and all(r['completion_ids'] for r in saved)
        save(out / 'replies.json', saved, immutable=True)
        memory = __import__('subprocess').check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True)
        report = dict(generation_seconds=elapsed, load_seconds=load_seconds,
                      tokens=sum(len(r['completion_ids']) for r in saved),
                      requests=len(saved), warmup_timed_token_repeat=all(repeat),
                      process_memory_at_generation_end=memory,
                      torch_main_process_peak_bytes=torch.cuda.max_memory_allocated(),
                      memory_note='vLLM engine runs in a child; nvidia-smi process memory is recorded separately from PyTorch parent peak')
        report['tokens_per_second'] = report['tokens'] / elapsed
        report['device_peak_used_mib'] = memory_monitor.finish()
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        device = next(g for g in metadata['gpu'] if g['index'] == visible or g['uuid'] == visible)
        report['selected_gpu_uuid'] = device['uuid']
        report['selected_gpu_peak_used_mib'] = report['device_peak_used_mib'][device['uuid']]
        save(out / 'performance.json', report)
    finally:
        engine.llm_engine.engine_core.shutdown()


def score(out):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from telemetry import Metrics
    config = load(out / 'registration.json')['config']
    dtype = torch.bfloat16 if config['dtype'] == 'bfloat16' else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True,
                                               dtype=dtype, attn_implementation='sdpa').to('cuda').eval()
    if config['dtype'] == 'float32':
        sys.path.insert(0, str(FAST))
        from precision import selected_logprobs
    else:
        sys.path.insert(0, str(PARENT))
        from train import selected_logprobs
    errors = []
    scored = []
    start = time.perf_counter()
    with torch.no_grad():
        for row in load(out / 'replies.json'):
            lp = selected_logprobs(model, row, .7).cpu()
            old = torch.tensor(row['behavior_logprobs'])
            assert len(lp) == len(old) and torch.isfinite(lp).all()
            errors.extend((lp - old).abs().tolist())
            scored.append(lp.tolist())
    assert errors
    gate = load(HERE / 'PLAN.json')['parity_gate']
    report = dict(mean_abs=float(np.mean(errors)), p99_abs=float(np.quantile(errors, .99)),
                  max_abs=float(np.max(errors)), tokens=len(errors),
                  hf_seconds=time.perf_counter() - start,
                  hf_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  hf_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    report['passed'] = (report['mean_abs'] <= gate['mean_absolute_sampled_logprob_error_max']
                        and report['p99_abs'] <= gate['p99_absolute_sampled_logprob_error_max']
                        and report['max_abs'] <= gate['max_absolute_sampled_logprob_error_max'])
    save(out / 'hf_logprobs.json', scored, immutable=True)
    save(out / 'parity.json', report, immutable=True)
    tracker = Metrics(out / 'telemetry', 'candidate/' + out.name, config)
    tracker.log(dict(optimizer_step=0, **{'parity/' + k: report[k] for k in ['mean_abs', 'p99_abs', 'max_abs', 'passed']},
                     **{'benchmark/' + k: v for k, v in load(out / 'performance.json').items()
                        if isinstance(v, (int, float))}))
    tracker.table('candidate', [{'config': config['name'], **report, **load(out / 'performance.json')}])
    tracker.finish(report['passed'])
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'sample', 'score'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--prompts', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out)
    elif args.stage == 'sample':
        sample(args.config, args.prompts, args.out)
    else:
        score(args.out)
