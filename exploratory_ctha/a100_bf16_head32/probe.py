"""Conditional fresh-trajectory BF16-head32 probe; never initializes a learning arm."""
import argparse
import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'a100_full_response'
FAST = HERE.parent / 'v3_fp32_fast'
OUT = ROOT / '.runtime/runs/a100_bf16_head32_r1'
sys.path[:0] = [str(OLD), str(HERE.parent / 'a100_validation')]
import a100_common
from a100_common import MODEL, fingerprint, hardware, load, save, sha


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in [*HERE.glob('*.py'), HERE / 'PLAN.json',
        OLD / 'trainer.py', OLD / 'checkpoint_live.py', OLD / 'direction_policy.py',
        HERE.parent / 'train.py', HERE.parent / 'habits_v2.py', FAST / 'collect.py', FAST / 'PLAN.json']}


def configure():
    sys.path.insert(0, str(FAST))
    import util
    util.DATA, util.MODEL, util.source_hashes = OUT, MODEL, sources
    return util


def legacy(name):
    spec = importlib.util.spec_from_file_location('bf16_' + name, FAST / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Policy:
    def __init__(self, module, adapter=None):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        self.sampler = module.SamplingPolicy.sample
        self.llm = LLM(model=str(MODEL), dtype='bfloat16', hf_overrides={'head_dtype': 'float32'},
            seed=0, max_model_len=6144, gpu_memory_utilization=.4, max_num_seqs=16,
            enforce_eager=True, enable_prefix_caching=True, enable_lora=bool(adapter),
            max_lora_rank=16, lora_dtype='bfloat16', generation_config='vllm', logprobs_mode='processed_logprobs')
        self.tok, self.temp = self.llm.get_tokenizer(), .7
        self.lora = LoRARequest('bf16_probe', 1, str(adapter)) if adapter else None

    def sample(self, prompts, seeds):
        return self.sampler(self, prompts, seeds)

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['collect', 'prepare', 'train', 'checkpoint', 'updated_sample', 'updated_score'])
    a = p.parse_args()
    assert load(OUT / 'verification.json')['passed'], 'Original numeric gate must pass before fresh probe'
    if not (OUT / 'phase2_registration.json').exists():
        assert a.action == 'collect'
        save(OUT / 'phase2_registration.json', dict(time=time.time(), source_sha256=sources(),
            inventory=hardware(), optimizer=load(FAST / 'PLAN.json')['optimizer'],
            probe='Eight real balanced advantage responses, four updates, original gates; no learning arm'), immutable=True)
    assert load(OUT / 'phase2_registration.json')['source_sha256'] == sources()
    util = configure()
    if a.action == 'collect':
        module = legacy('collect')
        Original = module.SamplingPolicy

        class CollectorPolicy(Policy):
            def __init__(self, *args):
                module.SamplingPolicy = Original
                super().__init__(module)

        original = module.provenance

        def provenance(args, games):
            value = original(args, games)
            value.pop('fingerprint')
            value.update(schema='ctha-v3-bf16-head32', precision='BF16_backbone_FP32_head',
                bf16_protocol_sha256=sha(HERE / 'PLAN.json'), script=str(Path(__file__).relative_to(ROOT)),
                gpu_inventory=hardware(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            value['fingerprint'] = fingerprint(value)
            return value

        module.provenance = provenance
        module.SamplingPolicy = CollectorPolicy
        sys.argv = ['collect.py', '--model', str(MODEL), '--games', str(ROOT / '.runtime/runs/ctha_a100_small_baseline_r1/train_games.json'),
            '--split', 'train', '--iteration', '0', '--out', str(OUT / 'rollouts'), '--reps', '4',
            '--workers', '4', '--chunk-games', '4', '--max-steps', '50', '--max-model-len', '6144',
            '--temperature', '.7', '--gpu-util', '.4']
        module.main()
    elif a.action == 'prepare':
        from habits import load_episodes
        from habits_v2 import action_base_advantages
        episodes = load_episodes(OUT / 'rollouts', 0)
        assert len(episodes) == 16
        advantages, _, _ = action_base_advantages(episodes, base='grpo', gamma=1., local_weight=1.)
        rows = []
        for e in episodes:
            for i, step in enumerate(e['steps']):
                rows.append(dict(prompt=step['prompt'], prompt_token_ids=step['prompt_token_ids'],
                    response_token_ids=step['response_token_ids'], behavior_logprobs=step['behavior_logprobs'],
                    advantage=advantages[e['episode_id'], i], episode_id=e['episode_id'], turn=step['turn']))
        dest = OUT / 'prepared'
        dest.mkdir(exist_ok=False)
        (dest / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        save(dest / 'advantage_report.json', dict(samples_sha256=sha(dest / 'samples.jsonl'),
            all_rows=len(rows), zero_advantage_rows=sum(r['advantage'] == 0 for r in rows), auxiliary=0), immutable=True)
    elif a.action == 'train':
        from telemetry import Metrics
        # The actor loop/loss remains byte-equivalent in AST to R3; only dtype/head/schema differ.
        import ast
        def loop(path):
            function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            return ast.dump(next(n for n in function.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch'))
        assert loop(HERE / 'trainer.py') == loop(OLD / 'trainer.py')
        spec = importlib.util.spec_from_file_location('bf16_head32_trainer', HERE / 'trainer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.train(SimpleNamespace(data=OUT / 'prepared/samples.jsonl', summary=OUT / 'rollouts/summary.json',
            out=OUT / 'direction_probe', adapter=None, seed=0, probe=True))
    elif a.action == 'checkpoint':
        # Preserve the original CPU checker, changing only the backbone load dtype/head setup.
        import torch
        from transformers import AutoModelForCausalLM
        from bf16_precision import setup_head
        original_load = AutoModelForCausalLM.from_pretrained
        def load_bf16(*args, **kw):
            kw['dtype'] = torch.bfloat16
            model = original_load(*args, **kw)
            setup_head(model)
            return model
        AutoModelForCausalLM.from_pretrained = staticmethod(load_bf16)
        runpy.run_path(str(HERE.parent / 'v3_engineering_fast_20261113/check_checkpoint.py'), run_name='__main__')
    elif a.action == 'updated_sample':
        module = legacy('collect')
        policy = Policy(module, OUT / 'direction_probe')
        try:
            rows = [json.loads(l) for l in (OUT / 'prepared/samples.jsonl').read_text().splitlines()]
            rows = [rows[i] for i in load(OUT / 'direction_probe/probe_indices.json')]
            replies = policy.sample([r['prompt'] for r in rows], [1000 + i for i in range(8)])
            save(OUT / 'updated_replies.json', [dict(prompt_ids=s['prompt_token_ids'], completion_ids=s['token_ids'],
                behavior_logprobs=s['behavior_logprobs']) for group in replies for s in group], immutable=True)
        finally:
            policy.close()
    else:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        from bf16_precision import setup_head, selected_logprobs
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa').to('cuda')
        setup_head(model)
        model = PeftModel.from_pretrained(model, str(OUT / 'direction_probe')).eval()
        scores, errors = [], []
        with torch.no_grad():
            for row in load(OUT / 'updated_replies.json'):
                lp = selected_logprobs(model, row, .7).cpu()
                scores.append(lp.tolist())
                errors.extend((lp - torch.tensor(row['behavior_logprobs'])).abs().tolist())
        save(OUT / 'updated_hf_logprobs.json', scores, immutable=True)
        report = dict(mean_abs=float(np.mean(errors)), p99_abs=float(np.quantile(errors, .99)), max_abs=float(np.max(errors)))
        report['passed'] = report['mean_abs'] <= .01 and report['p99_abs'] <= .05 and report['max_abs'] <= .25
        save(OUT / 'updated_verification.json', report, immutable=True)
        assert report['passed'], 'Native BF16 LoRA updated parity failed'


if __name__ == '__main__':
    main()
