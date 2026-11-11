"""Explicit hardware-resume overlay; frozen sampling/prompt/rollout body retained.
Only resume metadata comparison and new-record hardware receipt differ from run().
"""
import argparse,json,os,sys,subprocess,time
from pathlib import Path
import four_arm_eval as R
from four_arm_eval import CONDITIONS,CarryState,forced_release_action,prompt_for,common,E
import server_eval as S
H=Path(__file__).resolve().parent;D=H/'resume_20261111'

def compatible_provenance(a,b):
    # Git commit is audit metadata; source hashes, argv and every science field
    # must remain exact. Actual new commits are retained in new records.
    return isinstance(a,dict) and {k:v for k,v in a.items() if k != 'commit'}=={k:v for k,v in b.items() if k != 'commit'}

def runtime_receipt():
    p=json.loads((D/'registration.json').read_text())
    return dict(registration_sha256=E.file_sha256(D/'registration.json'),gpu_inventory=p['new_gpu_inventory'],runner_sha256=E.file_sha256(__file__),original_scientific_source_sha256=E.file_sha256(R.__file__))

def check_registration():
    p=json.loads((D/'registration.json').read_text())
    for path,sha in p['source'].items():assert E.file_sha256(path)==sha,path
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    assert inventory==p['new_gpu_inventory']
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['new_gpu_uuid']
    return p

def run(games, reps, pol, pool, out_dir, cond, alfenv, prov):
    """Uses E4 seeds, 50-step cap and exact same games; only prompt/decoding differs."""
    if cond not in CONDITIONS:
        raise ValueError(cond)
    from e4_01_rollout import seed_for

    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for g in games:
        for r in range(reps):
            path = Path(out_dir, f"s{g['seed']}_r{r}.json")
            if path.exists():
                old = json.loads(path.read_text())
                if old.get("condition") != cond or not compatible_provenance(old.get("provenance"), prov):
                    raise RuntimeError(f"Completed output has different configuration: {path}")
            else:
                jobs.append((g, r))
    if not jobs:
        return
    keys = [f"{g['seed']}:{rep}" for g, rep in jobs]
    opened = pool.open([(key, os.path.join(alfenv.ALFWORLD_DATA, g["game_file"]), [], None)
                        for key, (g, _) in zip(keys, jobs)])
    active_state = {}
    for key, (g, rep) in zip(keys, jobs):
        obs, adm, won, div = opened[key]
        if div:
            raise RuntimeError(f"Unexpected prefix divergence on reset: {key}")
        cs = CarryState()
        cs.update(obs)
        active_state[key] = dict(seed=g["seed"], rep=rep, game_file=g["game_file"],
                                 task_type=g["task_type"], task=common.extract_task(obs),
                                 initial_obs=obs, obs=obs, adm=adm, mem=[], steps=[],
                                 carry=cs, won=bool(won), done=bool(won))
    saved = set()
    def save_finished(key):
        if key in saved:
            return
        x = active_state[key]
        result = {k: v for k, v in x.items() if k not in ("carry", "obs", "adm", "mem")}
        result.update(n_turns=len(x["steps"]), condition=cond, provenance=prov, hardware_resume=runtime_receipt())
        E.write_json(os.path.join(out_dir, f"s{x['seed']}_r{x['rep']}.json"), result)
        saved.add(key)
    started = time.monotonic()
    try:
        for turn in range(1, common.MAX_STEPS + 1):
            active = [key for key in keys if not active_state[key]["done"]]
            if not active:
                break
            prompts = [prompt_for(active_state[key]["task"], active_state[key]["mem"],
                                  active_state[key]["obs"], active_state[key]["adm"],
                                  active_state[key]["carry"], cond) for key in active]
            seeds = [seed_for(active_state[key]["seed"], turn, active_state[key]["rep"]) for key in active]
            generation_start = time.monotonic()
            model_out = pol.sample(prompts, seeds, logprobs=False)
            generation_seconds = time.monotonic() - generation_start
            to_execute = {}
            for key, prompt, sample_seed, out in zip(active, prompts, seeds, model_out):
                x = active_state[key]
                proposed, valid, fmt = common.parse(out[0]["text"])
                forced = (forced_release_action(x["carry"], x["task"], x["task_type"], x["adm"])
                          if cond in ("hard", "both") else None)
                chosen = forced or proposed
                if forced is not None and forced not in x["adm"]:
                    raise RuntimeError("Recovery rule chose a non-admissible action")
                to_execute[key] = chosen
                x["steps"].append(dict(turn=turn, action=chosen, model_action=proposed,
                                        rule_triggered=forced is not None,
                                        overridden=forced is not None and proposed != forced,
                                        held_before=x["carry"].item if x["carry"].known else None,
                                        held_known_before=x["carry"].known,
                                        valid=True if forced is not None else valid,
                                        model_valid=valid, fmt=fmt, response=out[0]["text"], sample_seed=sample_seed,
                                        obs_before=x["obs"], admissible=x["adm"]))
            environment_start = time.monotonic()
            responses = pool.step(to_execute)
            for key in active:
                x = active_state[key]
                obs, adm, won, done = responses[key]
                x["carry"].update(obs)
                x["steps"][-1].update(obs_after=obs, won_after=bool(won),
                                       held_after=x["carry"].item if x["carry"].known else None,
                                       held_known_after=x["carry"].known)
                x["mem"].append(dict(text_obs=x["obs"], action=to_execute[key]))
                x["obs"], x["adm"] = obs, adm
                x["won"], x["done"] = bool(won), bool(won or done)
                if x["done"]:
                    save_finished(key)
            print(f"[{cond}] turn={turn} active={len(active)} won={sum(active_state[k]['won'] for k in keys)}", flush=True)
            print(json.dumps(dict(event="turn_timing", condition=cond, turn=turn, active=len(active),
                                  generation_seconds=generation_seconds,
                                  environment_seconds=time.monotonic()-environment_start,
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
        for key in keys:
            save_finished(key)
    finally:
        pool.close(keys)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a,rest=ap.parse_known_args()
    registration=check_registration();plan=json.loads(a.plan.read_text())
    for path,sha in plan['source'].items():assert E.file_sha256(path)==sha,path
    assert all(os.environ.get(k)==v for k,v in plan['environment'].items())
    # Only the preregistered isolated compatibility pilot may run before the gate.
    out=rest[rest.index('--out')+1] if '--out' in rest else str(H/'data/rollout')
    if Path(out).resolve()!=D/'pilot':
        gate=json.loads((D/'compatibility.json').read_text());assert gate['passed'] and gate['episodes']==48
        assert gate['registration_sha256']==E.file_sha256(D/'registration.json')
    else:
        assert rest[rest.index('--reps')+1]=='1'
        assert rest[rest.index('--policy-tag')+1] in plan['pilot_policies']
    os.environ['E4E_EXECUTION_PLAN_SHA256']=E.file_sha256(a.plan)
    # Completed summaries are immutable too; validate identical statistics and
    # retain their original provenance rather than rewriting after a Git commit.
    original_write=E.write_json
    def preserve_summary(path,value):
        dest=Path(path)
        if dest.name=='summary.json' and dest.exists():
            old=json.loads(dest.read_text())
            assert {k:v for k,v in old.items() if k!='provenance'}=={k:v for k,v in value.items() if k!='provenance'},str(dest)
            assert compatible_provenance(old['provenance'],value['provenance']),str(dest)
            return
        original_write(path,value)
    E.write_json=preserve_summary
    S.install(plan['engine_changes']);R.run=run
    sys.argv=[str(H/'four_arm_eval.py'),*rest];R.main()

if __name__=='__main__':main()
