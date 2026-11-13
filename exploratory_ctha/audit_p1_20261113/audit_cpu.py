"""Read-only P1 implementation audit. CPU checks are not benchmark results."""
import collections, contextlib, hashlib, json, math, pathlib, sys
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'exploratory_ctha'))
import train
import torch
DATA = pathlib.Path('/root/autodl-tmp/ctha_v2_tokenfix')

def load(p): return json.loads(p.read_text())

def gradient(advantage, ratio=1.0):
    lp=torch.tensor(math.log(ratio), dtype=torch.float64, requires_grad=True)
    loss=train.clipped_token_loss(lp.reshape(1),torch.zeros(1,dtype=torch.float64),advantage,[True],0.2)
    loss.backward()
    return lp.grad.item()

checks={}
assert gradient(1)<0 and gradient(-1)>0
checks['surrogate_gradient_direction']='passed: positive advantage raises sampled logprob; negative lowers it'
assert gradient(1,1.3)==0 and gradient(-1,.7)==0
checks['ppo_clip_flat_regions']='passed'

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.zeros(1))
    def forward(self,input_ids,use_cache):
        # Prompt length 2, continuation [2,3]. Correct causal positions are 1,2.
        logits=torch.zeros(1,4,5);logits[0,1,2]=4;logits[0,2,3]=4
        return type('Result',(),{'logits':logits})()
with patch('torch.cuda.is_bf16_supported',return_value=False),patch('torch.autocast',side_effect=lambda *a,**k:contextlib.nullcontext()):
    lp=train.selected_logprobs(Tiny(),{'prompt_ids':[0,1],'completion_ids':[2,3]},1.)
assert torch.allclose(lp,torch.full((2,),4-math.log(math.exp(4)+4)))
checks['causal_token_alignment']='passed on CPU with deterministic mock logits'

# Demonstrate why dropping zero rows BEFORE forming minibatches is not exact compute skipping.
full=gradient(1)/8
filtered=gradient(1)
assert filtered/full==8
checks['zero_filter_counterexample']={'one_nonzero_plus_seven_zero_rows_gradient':full,'filter_then_normalize_gradient':filtered,'factor':8.,'interpretation':'not equivalent compute-only optimization; actual multi-minibatch Adam effect is not this simple scalar'}

def adam(reset):
    x=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64));opt=torch.optim.AdamW([x],lr=.1,weight_decay=0)
    for i,g in enumerate([1.,1.,1.,-1.]):
        if reset and i==3:opt=torch.optim.AdamW([x],lr=.1,weight_decay=0)
        opt.zero_grad();x.grad=torch.tensor(g,dtype=torch.float64);opt.step()
    return x.item()
continuous,reset=adam(False),adam(True)
assert abs(continuous-reset)>.01
checks['optimizer_reset_counterexample']={'continuous_final_parameter':continuous,'reset_at_round_boundary_final_parameter':reset,'interpretation':'same incoming parameters and next gradient, different update due to lost moments/step counter'}

arms={}
for name in ['grpo','gigpo','fixed','ctha','progress']:
    rounds=[]
    for d in sorted((DATA/(name+'_s0')).glob('round_*')):
        t=load(d/'adapter/train_log.json');a=load(d/'prepared/advantage_report.json')
        rounds.append({'round':d.name,'steps':t['n_source_steps'],'nonzero_steps':t['n_nonzero_advantage_steps'],'updates':t['optimizer_steps'],'aux_samples':a['nonzero_aux_samples'],'gate':a['gate']})
    arms[name]={'rounds':rounds,'steps':sum(x['steps'] for x in rounds),'nonzero_steps':sum(x['nonzero_steps'] for x in rounds),'updates':sum(x['updates'] for x in rounds)}
initial={}
for name in ['gigpo','fixed','ctha','progress']:
    c=collections.Counter()
    for p in sorted((DATA/'grpo_s0/round_000/rollouts').glob('s*_r*.json')):
        a=load(p);b=load(DATA/(name+'_s0')/'round_000/rollouts'/p.name)
        assert a['provenance']['adapter_sha256'] is None and b['provenance']['adapter_sha256'] is None
        assert a['provenance']==b['provenance']
        x,y=a['steps'][0],b['steps'][0]
        assert x['prompt']==y['prompt'] and x['sample_seed']==y['sample_seed']
        c['episodes']+=1;c['different_first_reply']+=x['response_token_ids']!=y['response_token_ids'];c['different_first_action']+=x['action']!=y['action'];c['different_final_outcome']+=a['won']!=b['won']
    initial[name]=dict(c)
games=load(ROOT/'exploratory_E4/games.json')
assert not {x['game_file'] for x in games['train']}&{x['game_file'] for x in games['tune']}
report={'cpu_only':True,'no_training_or_old_results_modified':True,'checks':checks,'arms':arms,'same_base_first_round_vs_grpo':initial,'tune_task_types':dict(collections.Counter(x['task_type'] for x in games['tune'])),'source_sha256':{n:hashlib.sha256((ROOT/'exploratory_ctha'/n).read_bytes()).hexdigest() for n in ['train.py','run.py','rollout.py','prepare.py','habits_v2.py']},'limits':['No new GPU forward or model update was performed.','No unique cause of flat learning is established.','Recomputed HF old logprobs are not directly checked against actual vLLM sampling logprobs, which were not recorded.']}
(HERE/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':checks,'initial_replication':initial},indent=2))
