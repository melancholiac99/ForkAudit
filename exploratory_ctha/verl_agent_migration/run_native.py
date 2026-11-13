"""Bounded, isolated entrypoint; never invokes the upstream 150-epoch script."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def start_memory_monitor(output, devices):
    import csv
    import pynvml
    pynvml.nvmlInit()
    handles = [(i, pynvml.nvmlDeviceGetHandleByIndex(i)) for i in devices]
    hardware = [dict(index=i, uuid=str(pynvml.nvmlDeviceGetUUID(h)),
        memory_bytes=pynvml.nvmlDeviceGetMemoryInfo(h).total) for i, h in handles]
    stop = threading.Event()

    def sample():
        with (output / 'gpu_memory.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['unix_time', 'gpu_index', 'gpu_uuid', 'used_mib'])
            while not stop.is_set():
                for i, handle in handles:
                    writer.writerow([time.time(), i, str(pynvml.nvmlDeviceGetUUID(handle)),
                                     pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2])
                stream.flush()
                stop.wait(.2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return stop, thread, hardware

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--algorithm', choices=['gigpo', 'grpo'], default='gigpo')
    parser.add_argument('--updates', type=int, choices=[1, 4], default=1)
    parser.add_argument('--correction', choices=['native', 'bypass_ppo_clip'], default='native')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gpus', type=int, choices=[1, 2], default=2)
    parser.add_argument('--device', type=int, choices=[0, 1])
    parser.add_argument('--frozen-base-bf16', action='store_true')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--val-only', action='store_true')
    parser.add_argument('--final-eval', action='store_true')
    parser.add_argument('--max-num-seqs', type=int, choices=[1, 16], default=16)
    parser.add_argument('--reference-packet', type=Path)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--micro-batch', type=int, choices=[16, 32], default=32)
    parser.add_argument('--ray-cpus', type=int, choices=[8, 12, 16, 24], default=24)
    parser.add_argument('--probe-backend', choices=['hf', 'vllm'], default='hf')
    parser.add_argument('--gradient-bench-packet', type=Path)
    parser.add_argument('--logprob-micro', type=int, choices=[16, 32])
    parser.add_argument('--model-seed', type=int)
    args = parser.parse_args()
    assert Path(sys.prefix).resolve() == (ROOT / '.runtime/venvs/verl-agent-20bd331b').resolve()
    assert Path(args.name).name == args.name
    output = ROOT / '.runtime/runs' / args.name
    output.mkdir(parents=True, exist_ok=False)
    source = Path(os.environ['VERL_AGENT_SOURCE'])
    env = os.environ.copy()
    env.pop('FORKAUDIT_REFERENCE_PACKET', None)
    env.pop('FORKAUDIT_GRADIENT_BENCH_PACKET', None)
    env.pop('FORKAUDIT_MODEL_SEED', None)
    if args.model_seed is not None:
        assert args.frozen_base_bf16, 'Seed hook requires registered external_lib'
        env['FORKAUDIT_MODEL_SEED'] = str(args.model_seed)
    if args.device is not None:
        assert args.gpus == 1
        env['CUDA_VISIBLE_DEVICES'] = str(args.device)
    env.update(PYTHONPATH=f'{HERE}:{source}', FORKAUDIT_VERL_TASKS=str(ROOT / '.runtime/verl-agent-setup/fixed_tasks.json'),
               FORKAUDIT_VERL_OUTPUT=str(output), FORKAUDIT_VERL_CORRECTION=args.correction,
               TENSORBOARD_DIR=str(ROOT / '.runtime/tensorboard/ctha-a100-r2' / args.name),
               VLLM_ATTENTION_BACKEND='FLASH_ATTN', HYDRA_FULL_ERROR='1')
    env['RAY_TMPDIR'] = '/tmp/fa_' + hashlib.sha256(args.name.encode()).hexdigest()[:10]
    if args.reference_packet:
        assert args.val_only and args.resume_from and args.gpus == 1
        env['FORKAUDIT_REFERENCE_PACKET'] = str(args.reference_packet.resolve())
    if args.gradient_bench_packet:
        assert args.reference_packet and args.val_only and args.resume_from
        env['FORKAUDIT_GRADIENT_BENCH_PACKET'] = str(args.gradient_bench_packet.resolve())
    data = output / 'data'
    # Upstream uses geometry3k only as four modality/index rows; all RL observations
    # and rewards come from the real ALFWorld environments checked separately.
    prep_env = dict(env, HF_HUB_OFFLINE='0', HF_DATASETS_OFFLINE='0')
    subprocess.run([sys.executable, '-m', 'examples.data_preprocess.prepare', '--mode', 'text',
        '--train_data_size', '4', '--val_data_size', '4', '--local_dir', str(data)],
        env=prep_env, cwd=source, check=True)
    settings = {
        'algorithm.adv_estimator': args.algorithm,
        'data.train_files': str(data / 'text/train.parquet'),
        'data.val_files': str(data / 'text/test.parquet'),
        'data.train_batch_size': 4, 'data.val_batch_size': 4,
        'data.max_prompt_length': 2048, 'data.max_response_length': 512,
        'data.filter_overlong_prompts': True, 'data.truncation': 'error', 'data.return_raw_chat': True,
        'actor_rollout_ref.model.path': os.environ['M0_MODEL'],
        'actor_rollout_ref.model.lora_rank': 64, 'actor_rollout_ref.model.lora_alpha': 64,
        'actor_rollout_ref.model.external_lib': 'checkpoint_evidence' if args.frozen_base_bf16 or args.save_checkpoint or args.resume_from else 'null',
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.optim.lr': 3e-6,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 256,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': args.micro_batch,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.actor.use_invalid_action_penalty': True,
        'actor_rollout_ref.actor.invalid_action_penalty_coef': .1,
        'actor_rollout_ref.rollout.name': args.probe_backend if args.reference_packet else 'vllm',
        'actor_rollout_ref.rollout.load_format': 'safetensors',
        'actor_rollout_ref.rollout.layered_summon': False,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': args.logprob_micro or args.micro_batch,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': args.gpus,
        'actor_rollout_ref.rollout.max_num_seqs': args.max_num_seqs,
        'actor_rollout_ref.rollout.gpu_memory_utilization': .5 if args.gpus == 1 else .6,
        'actor_rollout_ref.rollout.enable_chunked_prefill': False,
        'actor_rollout_ref.rollout.enforce_eager': args.eager,
        'actor_rollout_ref.rollout.free_cache_engine': False,
        'actor_rollout_ref.rollout.val_kwargs.temperature': .4,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 32,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'algorithm.use_kl_in_reward': False, 'algorithm.gamma': .95,
        'algorithm.gigpo.step_advantage_w': 1., 'algorithm.gigpo.mode': 'mean_std_norm',
        'env.env_name': 'alfworld/AlfredTWEnv', 'env.seed': 0, 'env.max_steps': 50,
        'env.rollout.n': 4, 'env.resources_per_worker.num_cpus': .1,
        'env.alfworld.eval_dataset': 'train',
        'trainer.critic_warmup': 0, 'trainer.logger': ['console', 'tensorboard'],
        'trainer.project_name': 'forkaudit_native_alfworld', 'trainer.experiment_name': args.name,
        'trainer.n_gpus_per_node': args.gpus, 'trainer.nnodes': 1,
        'trainer.save_freq': args.updates if args.save_checkpoint or args.updates > 1 else -1,
        'trainer.test_freq': args.updates if args.updates > 1 or args.final_eval else -1, 'trainer.total_epochs': args.updates,
        'trainer.total_training_steps': args.updates, 'trainer.val_before_train': args.updates > 1 or args.val_only,
        'trainer.val_only': args.val_only,
        'trainer.resume_mode': 'resume_path' if args.resume_from else 'disable',
        'trainer.resume_from_path': str(args.resume_from.resolve()) if args.resume_from else 'null',
        'trainer.default_local_dir': str(output / 'checkpoints'),
        'trainer.rollout_data_dir': str(output / 'rollouts'),
        'trainer.validation_data_dir': str(output / 'validation'),
        'ray_init.num_cpus': args.ray_cpus, '+ray_init.object_store_memory': 2147483648,
        '+ray_init.dashboard_port': 0,
        '+ray_init.runtime_env.worker_process_setup_hook': 'launch.setup_worker',
        'hydra.run.dir': str(output / 'hydra'),
    }
    if args.frozen_base_bf16:
        settings['+actor_rollout_ref.actor.fsdp_config.model_dtype'] = 'bf16'
    overrides = []
    for key, value in settings.items():
        rendered = json.dumps(value, separators=(',', ':')) if isinstance(value, (bool, list)) else str(value)
        overrides.append(f'{key}={rendered}')
    command = [sys.executable, str(HERE / 'launch.py'), *overrides]
    payload = dict(command=command, env={k: env[k] for k in env if k.startswith(('FORKAUDIT_', 'VLLM_', 'TENSORBOARD', 'CUDA_', 'RAY_'))},
                   upstream='20bd331bdbc9026a5668e11362178e10ab7400c8', settings=settings)
    payload['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sorted(HERE.glob('*.py')), *sorted(HERE.glob('*PLAN.json')), ROOT / 'env/requirements-verl-agent.txt', ROOT / 'env/verl_agent.sh']}
    payload['python_development_header_sha256'] = hashlib.sha256(Path('/usr/include/python3.12/Python.h').read_bytes()).hexdigest()
    stop, monitor, hardware = start_memory_monitor(output, [args.device] if args.device is not None else list(range(args.gpus)))
    payload['hardware'] = hardware
    payload['memory_sampling_interval_seconds'] = .2
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload['configuration_sha256'] = fingerprint
    (output / 'registration.json').write_text(json.dumps(payload, indent=2) + '\n')
    # Keep executable sources matching this registration even if later work
    # adds another candidate. Never overwrite a previous run directory.
    for relative, expected in payload['source_sha256'].items():
        content = (ROOT / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
        snapshot = output / 'registered_source' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    print(json.dumps(dict(directory=str(output), fingerprint=fingerprint)), flush=True)
    if args.prepare_only:
        stop.set(); monitor.join()
        return
    try:
        with (output / 'run.log').open('w') as log:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        stop.set(); monitor.join()
    (output / 'exit.json').write_text(json.dumps(dict(returncode=result.returncode)))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
