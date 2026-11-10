"""Extend the frozen E4e queue with user-requested GiGPO under the HCAPO harness."""
import argparse,collections,hashlib,json,os,sys,time
from pathlib import Path
import run_server as C
V=C.V;H=C.H;G=H/'gigpo_20261110';OUT=H/'results_20261110/gigpo';MODEL=Path('/root/autodl-tmp/models/GiGPO-7B-ALFWorld')
_original_shutdown=C.shutdown
_original_notify=C.notify

def notify(message):
    if '【E4e已完成并发布】' in message:
        message='【E4e基础四臂已发布；GiGPO尚待完成】原E4e已核验上传，但用户追加的GiGPO评估还在本队列内，不能提前关机。'+message.replace('控制器将sync并关机','控制器将接着运行GiGPO，全部核验上传后才关机')
    if '【E4e单卡队列异常】' in message:
        message=message.replace('恢复入口 exploratory_E4e/run_server.py','恢复入口 exploratory_E4e/run_with_gigpo.py')
        message+=' 用户已指定下载并追加GiGPO完整模型；检查 exploratory_E4e/gigpo_20261110 的日志，必须保留这项任务。'
    message=message.replace('无本地公开RL权重，已明确列为缺项。','GiGPO公开RL模型已追加下载评测，检查其独立状态。').replace('公开RL比较缺本地权重，未训练。','GiGPO评测尚待完成，未训练。').replace('运行中的E4e控制器会汇总、上传credit-grok并关机','运行中的控制器会汇总并上传credit-grok，再完成GiGPO评测后关机')
    _original_notify(message)
C.notify=notify

def check_extension():
    p=V.load(G/'extension_policy.json')
    for file,sha in p['source'].items():assert V.sha(file)==sha,file
    assert p['shutdown_only_after_gigpo']
    return p

def prepare_model():
    C.state('waiting_gigpo_download');start=time.monotonic()
    while True:
        status=V.load(G/'download_status.json')
        if status['stage']=='completed_verified':break
        assert status['stage']!='failed','GiGPO download failed: '+str(status)
        assert time.monotonic()-start<10800,'GiGPO download did not finish within bounded wait'
        time.sleep(20)
    import download_gigpo as download
    files=download.verify()
    from transformers import AutoConfig,AutoTokenizer
    cfg=AutoConfig.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    assert cfg.model_type=='qwen2' and cfg.architectures==['Qwen2ForCausalLM']
    text=tok.apply_chat_template(V.R.common.chat_messages('Choose an admissible action.'),tokenize=False,add_generation_prompt=True)
    assert text and cfg.vocab_size>=len(tok)
    compatible=dict(passed=True,architecture=cfg.architectures,vocab_size=cfg.vocab_size,tokenizer_size=len(tok),
        history_length=V.R.common.HISTORY_LENGTH,prompt='ForkAudit HCAPO, two-step observation/action history',
        limitation='Different evaluation prompt/history from the GiGPO paper/model-card setup. Do not compare directly with published paper success rates. Training configurations also differ from E4 DPO models.',
        model_card='https://huggingface.co/langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld',revision=V.load(G/'hub_metadata.json')['sha'])
    V.save(G/'compatibility.json',compatible)
    (G/'model_card.txt').write_bytes((MODEL/'README.md').read_bytes())
    old=V.load(C.PLAN);p=dict(old)
    p.update(model=str(MODEL),policies={'gigpo7b':None},conditions=['base'],full_records=402,pilot_policies=[],
        model_files={path:info['sha256'] for path,info in files.items()},public_checkpoint=compatible,
        missing_optional_public_rl=None,extension_parent_plan_sha256=V.sha(C.PLAN))
    dest=G/'plan.json'
    if dest.exists():assert V.load(dest)==p
    else:V.save(dest,p)
    return p

def verify_and_analyze():
    plan=V.load(G/'plan.json');games={g['seed']:g for g in V.load(V.R.E.GAMES_PATH)['valid_unseen']}
    folder=H/'data/rollout/valid_unseen/gigpo7b/base';files=sorted(folder.glob('s*_r*.json'))
    assert len(files)==402;eps=[];manifest={};valid=total=0
    expected={(s,r) for s in plan['full_games'] for r in range(3)};seen=set()
    for path in files:
        e=V.load(path);key=(e['seed'],e['rep']);assert key in expected and key not in seen;seen.add(key)
        g=games[e['seed']];p=e['provenance']
        assert e['game_file']==g['game_file'] and e['task_type']==g['task_type'] and e['condition']=='base'
        assert p['model']==str(MODEL) and p['adapter'] is None and p['execution_plan_sha256']==V.sha(G/'plan.json')
        assert p['source_sha256']==plan['source'][str(H/'four_arm_eval.py')] and p['games_sha256']==V.sha(V.R.E.GAMES_PATH)
        assert e['n_turns']==len(e['steps'])<=50 and e['task']==V.R.common.extract_task(e['initial_obs'])
        obs=e['initial_obs'];carry=V.S.CarryState();carry.update(obs)
        for i,s in enumerate(e['steps'],1):
            action,ok,fmt=V.R.common.parse(s['response'])
            assert action==s['action']==s['model_action'] and ok==s['valid']==s['model_valid'] and fmt==s['fmt']
            assert not s['rule_triggered'] and not s['overridden'] and s['obs_before']==obs
            assert s['turn']==i and s['sample_seed']==V.seed_for(e['seed'],i,e['rep'])
            assert s['held_known_before']==carry.known and s['held_before']==(carry.item if carry.known else None)
            carry.update(s['obs_after']);assert s['held_known_after']==carry.known and s['held_after']==(carry.item if carry.known else None)
            if i<len(e['steps']):assert not s['won_after']
            obs=s['obs_after'];valid+=int(ok);total+=1
        assert e['won']==bool(e['steps'] and e['steps'][-1]['won_after'])
        eps.append(e);manifest[str(path.relative_to(C.ROOT))]=V.sha(path)
    assert seen==expected
    OUT.mkdir(exist_ok=True)
    comparisons={}
    for tag in ['base','aprime_s0','aprime_s1','aprime_s2']:
        other=[V.load(p) for p in (V.data_root('full')/tag/'base').glob('s*_r*.json')]
        assert len(other)==402;comparisons[tag]=V.F.paired_ci(other,eps)
    analysis=dict(policy='gigpo7b',episodes=402,games=134,reps=3,success=sum(e['won'] for e in eps)/402,
        funnel=V.F.funnel_for(eps),model_parse_valid_rate=valid/total if total else None,
        descriptive_gigpo_minus_other_pp=comparisons,compatibility=V.load(G/'compatibility.json'),
        warning='HCAPO behavior/compatibility evaluation only; not a reproduction of the paper. Cross-policy comparisons are not causal RL-versus-DPO training effects.')
    V.save(OUT/'analysis.json',analysis);V.save(OUT/'raw_manifest.json',manifest)
    groups=collections.defaultdict(list)
    for p in files:groups[V.load(p)['task_type']].append(p)
    chosen=[]
    for kind,ps in sorted(groups.items()):chosen+=sorted(ps,key=lambda p:hashlib.sha256(('GiGPO-replay:'+p.name).encode()).hexdigest())[:2]
    oldout=V.RESULT
    try:V.RESULT=OUT;V.replay_audit('pilot',chosen)
    finally:V.RESULT=oldout
    (OUT/'pilot_environment_replay.json').replace(OUT/'environment_replay.json')
    V.save(OUT/'verification.json',dict(passed=True,episodes=402,raw_manifest_sha256=V.sha(OUT/'raw_manifest.json'),analysis_sha256=V.sha(OUT/'analysis.json'),environment_replay_sha256=V.sha(OUT/'environment_replay.json')))
    lines=['GiGPO-7B-ALFWorld under ForkAudit HCAPO (2-step history)',f'134 games x 3 repeats = 402 episodes; success {100*analysis["success"]:.2f}%',analysis['warning'],
        'The initial E4e registration lacked local public RL weights. This explicitly requested supplement supplies that comparison; original E4e four-arm results remain unchanged.']
    for tag,v in comparisons.items():lines.append(f"GiGPO minus {tag}: {v['difference_pp']:+.4f} pp, 95% game-cluster CI {v['ci95_pp']}")
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    (H/'results_20261110/COMBINED_REPORT.txt').write_text('E4e four-arm audit plus user-requested GiGPO supplement.\nFour-arm results: E4e_REPORT.txt and full_analysis.json.\nPublic RL under HCAPO: gigpo/REPORT.txt and gigpo/analysis.json.\nGiGPO is now evaluated; the initial missing-checkpoint note is historical. No paper-rate comparison or causal RL/DPO claim.\n')

def finish_with_gigpo(outcome):
    if outcome!='available_E4e_scope_verified_uploaded':return _original_shutdown(outcome)
    check_extension();plan=prepare_model();C.state('gigpo_evaluation')
    args=[sys.executable,'-u',str(H/'server_eval.py'),'--plan',str(G/'plan.json'),'--split','valid_unseen','--policy-tag','gigpo7b',
        '--model-path',str(MODEL),'--conditions','base','--reps','3','--workers','20','--chunk','402']
    dest=H/'data/rollout/valid_unseen/gigpo7b/base'
    if not (dest/'summary.json').exists():C.invoke('gigpo_full',args)
    C.state('gigpo_verification');verify_and_analyze()
    import download_gigpo
    download_gigpo.verify()
    C.state('publishing_combined');receipt=C.publish();V.save(G/'publication.json',receipt)
    try:_original_notify(f'【E4e与GiGPO全部完成】请读 {H}/results_20261110/COMBINED_REPORT.txt 和 gigpo/REPORT.txt；完整结果已核验上传credit-grok，模型卡提示模板不同，不能直接比较论文成功率。控制器正在按用户授权关机。')
    except Exception:(G/'notification_error.txt').write_text('Completion notification failed; verified publication retained.')
    _original_shutdown('E4e_and_GiGPO_verified_uploaded')
C.shutdown=finish_with_gigpo

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',type=int);a=ap.parse_args();check_extension()
    if a.watch:C.watch(a.watch)
    else:
        handoff=V.load(G/'handoff.json');pid=handoff.get('child')
        C.state('waiting_handoff_child',child=pid)
        if pid:
            cmd=Path(f'/proc/{pid}/cmdline')
            while C.live(pid) and cmd.exists() and b'server_eval.py' in cmd.read_bytes():
                assert Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]==handoff['child_start_ticks'],'PID identity changed'
                time.sleep(15)
        C.main()
