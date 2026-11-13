"""Local-path launcher around unchanged FP32 implementations and objectives."""
import argparse
import importlib.util
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from a100_common import DATA, FAST, HERE, MODEL, PARENT, ROOT, PeakMemory, fingerprint, hardware, load, save, sha, sources


def legacy(name):
    spec = importlib.util.spec_from_file_location('a100_legacy_' + name, FAST / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure():
    sys.path.insert(0, str(FAST))
    import util
    util.MODEL = MODEL
    util.DATA = DATA
    util.source_hashes = sources
    from precision import check_environment
    check_environment()
    return util


class Policy:
    def __init__(self, module, config, model, adapter, max_model_len, temperature, gpu_util, out):
        from vllm import LLM
        from merged_policy import export_policy
        assert config['dtype'] == 'float32', 'BF16 was rejected by real-sample parity gates'
        assert max_model_len == 6144 and temperature == .7 and gpu_util == config['gpu_memory_utilization']
        os.environ['CTHA_A100_CONFIG_JSON'] = json.dumps(config, sort_keys=True)
        self.module = module
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.export = export_policy(model, adapter) if adapter else None
        self.monitor = PeakMemory().start()
        self.failed = False
        self.tokens = 0
        self.seconds = 0.
        options = dict(model=str(self.export or model), dtype='float32', seed=0,
                       gpu_memory_utilization=gpu_util, max_model_len=max_model_len,
                       enforce_eager=True, enable_prefix_caching=True, generation_config='vllm',
                       enable_lora=False, logprobs_mode='processed_logprobs',
                       max_num_seqs=config['max_num_seqs'])
        if config['worker'] == 'a100':
            options['worker_cls'] = 'exploratory_ctha.a100_validation.worker.A100Worker'
        try:
            self.llm = LLM(**options)
        except Exception:
            save(self.out / 'device_memory_failure.json', self.monitor.finish())
            raise
        self.tok = self.llm.get_tokenizer()
        self.temp = temperature
        self.lora = None

    def sample(self, prompts, seeds):
        start = time.perf_counter()
        try:
            result = self.module.SamplingPolicyOriginal.sample(self, prompts, seeds)
        except Exception:
            self.failed = True
            raise
        self.seconds += time.perf_counter() - start
        self.tokens += sum(len(sample['token_ids']) for candidates in result for sample in candidates)
        return result

    def close(self):
        try:
            self.llm.llm_engine.engine_core.shutdown()
        finally:
            peak = self.monitor.finish()
            save(self.out / 'device_performance.json', dict(device_peak_used_mib=peak,
                 visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), generated_tokens=self.tokens,
                 generation_seconds=self.seconds, tokens_per_second=self.tokens / self.seconds if self.seconds else 0.))
        if self.export and not self.failed:
            assert self.export.parent == DATA and self.export.name.startswith('sampling_export_')
            shutil.rmtree(self.export)


def collect(args, config):
    module = legacy('collect')
    module.SamplingPolicyOriginal = module.SamplingPolicy
    module.SamplingPolicy = lambda model, adapter, length, temperature, gpu_util: Policy(
        module, config, model, adapter, length, temperature, gpu_util, args.out)
    original_provenance = module.provenance

    def provenance(a, games):
        result = original_provenance(a, games)
        result.pop('fingerprint')
        result.update(script='exploratory_ctha/a100_validation/backend.py',
                      source_sha256=sha(Path(__file__)),
                      legacy_collector_source_sha256=sha(FAST / 'collect.py'),
                      a100_config=config, a100_config_fingerprint=fingerprint(config),
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=hardware())
        result['precision'] = 'float32_tf32x3_' + (f'stages{config["stages"]}' if config['worker'] == 'a100' else 'native')
        if config['worker'] == 'a100':
            result['worker_sha256'] = sha(HERE / 'worker.py')
        else:
            import inspect
            import vllm.model_executor.determinism.batch_invariant as native
            result['worker_sha256'] = sha(Path(inspect.getsourcefile(native.matmul_persistent)))
        result['fingerprint'] = fingerprint(result)
        return result

    module.provenance = provenance
    sys.argv = [str(FAST / 'collect.py'), '--model', str(MODEL), '--games', str(args.games),
                '--split', 'train', '--iteration', '0', '--out', str(args.out),
                '--reps', str(args.reps), '--workers', '16', '--chunk-games', '16',
                '--max-steps', '50', '--max-model-len', '6144', '--temperature', '0.7',
                '--gpu-util', str(config['gpu_memory_utilization'])]
    if args.adapter:
        sys.argv += ['--adapter', str(args.adapter)]
    module.main()


def train(args, util):
    from telemetry import Metrics
    tracker = Metrics(args.out.parent / 'training_telemetry', 'direction-probe/four-steps',
                      dict(scope='engineering_only', config=load(args.config), optimizer=load(FAST / 'PLAN.json')['optimizer']))
    original_save = util.save
    training_rows = []

    def instrumented_save(path, value):
        original_save(path, value)
        name = Path(path).name
        if name == 'parity.json':
            tracker.log(dict(optimizer_step=0, **{'parity/' + k: value[k] for k in ['mean_abs', 'p99_abs', 'max_abs', 'passed']}))
        elif name == 'progress.json':
            row = dict(optimizer_step=value['updates'], elapsed_s=value['elapsed_s'], **value['last'])
            training_rows.append(row)
            tracker.log(dict(optimizer_step=row['optimizer_step'], **{
                'train/loss': row['loss'], 'train/grad_norm': row['grad_norm'],
                'train/elapsed_s': row['elapsed_s'], 'train/batch_rows': row['batch_rows'],
                'stability/k3': row['sampled_behavior_k3'], 'stability/clip_fraction': row['clip_fraction'],
                'stability/reference_k3': row['sampled_reference_k3']}))
            tracker.table('training_steps', training_rows)
        elif name == 'direction_check.json':
            tracker.log(dict(optimizer_step=4, **{'direction/' + k: v for k, v in value.items()}))
        elif name == 'post_update.json':
            tracker.table('per_response_direction', value['direction'])

    util.save = instrumented_save
    module = legacy('trainer')
    original_logprobs = module.selected_logprobs
    base_scores = []

    def capture_base(model, example, temperature):
        value = original_logprobs(model, example, temperature)
        if len(base_scores) < 8 and not model.training:
            base_scores.append(dict(prompt_ids=example['prompt_ids'], completion_ids=example['completion_ids'],
                                    hf_logprobs=value.detach().cpu().tolist(),
                                    behavior_logprobs=example['old'].tolist()))
        return value

    module.selected_logprobs = capture_base
    monitor = PeakMemory().start()
    success = False
    try:
        module.train(SimpleNamespace(data=DATA / 'initial_prepared/samples.jsonl',
                     summary=DATA / 'initial_train/summary.json', out=args.out,
                     adapter=None, seed=0, probe=True))
        success = True
    finally:
        if base_scores:
            save(args.out / 'base_parity_replies.json', base_scores, immutable=True)
        save(DATA / 'probe_device_peak.json', monitor.finish())
        tracker.finish(success)


def adapter_sample(config):
    module = legacy('collect')
    module.SamplingPolicyOriginal = module.SamplingPolicy
    out = DATA / 'adapter_parity'
    out.mkdir(exist_ok=False)
    policy = Policy(module, config, MODEL, DATA / 'direction_probe', 6144, .7,
                    config['gpu_memory_utilization'], out)
    try:
        rows = [json.loads(line) for line in (DATA / 'initial_prepared/samples.jsonl').read_text().splitlines()]
        rows = [rows[i] for i in load(DATA / 'direction_probe/probe_indices.json')]
        result = policy.sample([r['prompt'] for r in rows], [1000 + i for i in range(len(rows))])
        replies = [dict(prompt_ids=sample['prompt_token_ids'], completion_ids=sample['token_ids'],
                        behavior_logprobs=sample['behavior_logprobs']) for candidates in result for sample in candidates]
        save(out / 'replies.json', replies, immutable=True)
    finally:
        policy.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'prepare', 'train', 'checkpoint', 'adapter_sample', 'adapter_score'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--games', type=Path)
    parser.add_argument('--reps', type=int, default=4)
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    config = load(args.config)
    assert config['dtype'] == 'float32'
    util = configure()
    if args.action == 'collect':
        collect(args, config)
    elif args.action == 'prepare':
        legacy('prepare_v3').prepare(DATA / 'initial_train', DATA / 'initial_prepared', base='grpo', mode='grpo')
    elif args.action == 'train':
        train(args, util)
    elif args.action == 'checkpoint':
        runpy.run_path(str(PARENT / 'v3_engineering_fast_20261113/check_checkpoint.py'), run_name='__main__')
    elif args.action == 'adapter_sample':
        adapter_sample(config)
    else:
        module = legacy('adapter_parity')
        original = module.selected_logprobs
        scores = []

        def capture_updated(model, row, temperature):
            result = original(model, row, temperature)
            scores.append(result.detach().cpu().tolist())
            return result

        module.selected_logprobs = capture_updated
        sys.argv = ['adapter_parity.py', 'score']
        try:
            module.main()
        finally:
            if scores:
                save(DATA / 'adapter_parity/hf_logprobs.json', scores, immutable=True)
