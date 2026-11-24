"""CPU-only crossed label/loss fits, paired on the original T_cal positions and supports."""
from hashlib import sha256
import numpy as np
import torch

from .first_command_tables import estimate, table

from .action_ending import locate, vocabulary
from .action_ending_tcal import oof
from .b_fit import load_data, weights, PARENT, SEED as FIT_SEED
from .b_metrics import Cluster
from .e25_offline import shrink
from .e25_paths import RUN as E
from .e25_projector import FitData
from .first_command_entropy import DEST,quantities
from .gap_cpu_actions import rows
from .gap_cpu_packet import cluster_mean, DRAWS, SEED
from .gap_score import RUN, STAGE_B as B
from .projection import VisibleProjection
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, digest_value, read, save

CACHE=RUN/'first_command_replay/crossed_fits'
ARMS=('full_huber','full_kl','content_huber','content_kl')
PAIRINGS={'full_kl_minus_huber':(1,0),'content_kl_minus_huber':(3,2),
          'huber_full_minus_content':(0,2),'kl_full_minus_content':(1,3)}


def huber_scale(pred,truth,mask,meta):
    coeff={}
    for c in sorted(set(meta['cls'])):
        i=meta['cls']==c
        p,y,m=pred[i],truth[i],mask[i]
        def average(x):return (x*m).sum(1)/m.sum(1)
        slope=Cluster(meta['goal'][i]).ratio(average(p*p),average(p*y),average(p),average(y))['raw']
        coeff[c]=float(np.clip(slope,0.,1.)) if slope is not None else 0.
    return coeff


def layout(meta,ids):
    tokens,_=vocabulary(); lookup={}
    for file in sorted(B.glob('tape_*.jsonl')):
        for x in rows(file):
            key=x['goal'],x['step'];ans=x['response']
            assert key not in lookup
            pos,_,_=locate([tokens[i] for i in ans])
            roles=['none']*len(ans)
            for p in pos:roles[p.index]='bracket' if p.role=='bracket' else 'following'
            lookup[key]=ans,roles
    pos=0; roles=[]
    while pos<len(ids):
        key=meta['goal'][pos],meta['step'][pos]
        ans,found=lookup.pop(key);n=len(ans)
        assert np.array_equal(ids[pos:pos+n,-1],ans) and np.all(meta['goal'][pos:pos+n]==key[0]) and np.all(meta['step'][pos:pos+n]==key[1])
        roles.extend(found);pos+=n
    assert not lookup
    return np.array(roles)


def diagnostics(pred,reference,mask):
    diff=(pred-reference)[mask].astype(float)
    return dict(max_abs=float(np.abs(diff).max()),rms=float(np.mean(diff**2)**.5))


def main():
    if torch.cuda.is_initialized():raise RuntimeError('CPU fit required')
    CACHE.mkdir(parents=True,exist_ok=True)
    ts,meta,_=load_data();idx=np.arange(len(meta['goal']))
    assert len(idx)==34825 and len(set(meta['goal']))==512
    feature_hash=sha256()
    for p in sorted((B/'features').glob('*.pt')):
        feature_hash.update((p.name+' '+digest_file(p)+'\n').encode())
    prior=read(DEST/'tcal_entropy.json')
    assert feature_hash.hexdigest()==prior['feature_digest']
    head=torch.load(B/'head.pt',weights_only=True,mmap=True,map_location='cpu')
    # Supply the two labels explicitly: Huber reads a centered vector, KL reads a distribution.
    content_teacher=(ts['s'].float()+ts['d'].float()).masked_fill(~ts['mask'],-torch.inf).softmax(-1)
    data=FitData(dict(ts,d=ts['old'],teacher=content_teacher),meta,head,'cpu')
    s=ts['s'].double().numpy();mask=data.mask.numpy();fold=meta['fold']; cls=meta['cls']
    roles=layout(meta,ts['ids'].numpy())
    originals,refs=oof([B/f'fold_{i}.npz' for i in range(4)],len(idx),'idx','original')
    contents,basefold=oof([B/f'fold_{i}.npz' for i in range(4)],len(idx),'idx','content')
    klprior,klfold=oof([E/f'offline/fold_{i}.npz' for i in range(4)],len(idx),'va','pred')
    assert np.array_equal(refs,fold) and np.array_equal(basefold,fold) and np.array_equal(klfold,fold)
    raw={'full_huber':np.empty(s.shape,np.float32),'content_kl':np.empty(s.shape,np.float32),
         'full_kl':np.empty(s.shape,np.float32),'content_huber':contents}
    manifest=PARENT/'asin_split_manifest.json'
    asin_by_goal={r['goal_index']:r['asin'] for r in read(manifest)['T_cal']}
    sources={str(p.relative_to(RUN.parent)):digest_file(p) for p in (
        manifest,B/'head.pt',B/'reports/calibration_strength_report.json',RUN/'initial/calibration.json')}
    receipts=[]
    for f in range(4):
        tr,va=idx[fold!=f],idx[fold==f]
        assert not (set(meta['goal'][tr])&set(meta['goal'][va]))
        assert {asin_by_goal[g] for g in meta['goal'][tr]}.isdisjoint(asin_by_goal[g] for g in meta['goal'][va])
        oldfit=read(B/f'fit_{f}.json')
        train_hash=digest_value(sorted(set(meta['goal'][tr].tolist())))
        test_hash=digest_value(sorted(set(meta['goal'][va].tolist())))
        assert train_hash==oldfit['train_ids_sha256'] and test_hash==oldfit['heldout_ids_sha256']
        for p in (B/f'fold_{f}.npz',E/f'offline/fold_{f}.npz',RUN/f'initial/fold_{f}.pt'):
            sources[str(p.relative_to(RUN.parent))]=digest_file(p)
        def predict(net):
            with torch.no_grad():
                return np.concatenate([data.predict(net,part).numpy() for part in np.array_split(va,(len(va)+511)//512)])
        net=VisibleProjection(head.shape[1])
        net.load_state_dict(torch.load(RUN/f'initial/fold_{f}.pt',weights_only=True,map_location='cpu'))
        raw['full_kl'][va]=predict(net.eval())
        for arm,loss in [('full_huber','huber'),('content_kl','kl')]:
            path=CACHE/f'{arm}_{f}.pt'
            receipt=CACHE/f'{arm}_{f}.json'
            if receipt.is_file():
                record=read(receipt)
                assert (record['arm'],record['fold'],record['seed'])==(arm,f,FIT_SEED+f)
                assert record['sha256']==digest_file(path) and record['feature_digest']==feature_hash.hexdigest()
                assert record['train_sha']==train_hash and record['test_sha']==test_hash
                net=VisibleProjection(head.shape[1])
                net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
            else:
                net=data.fit(tr,f,loss)
                torch.save(net.state_dict(),path)
                record=dict(arm=arm,fold=f,train_positions=len(tr),heldout_positions=len(va),train_tasks=len(set(meta['goal'][tr])),
                            heldout_tasks=len(set(meta['goal'][va])),train_sha=train_hash,test_sha=test_hash,sha256=digest_file(path),
                            feature_digest=feature_hash.hexdigest(),seed=FIT_SEED+f)
                save(receipt,record)
            raw[arm][va]=predict(net.eval())
            receipts.append(record)
            print(arm,'fold',f,'complete',flush=True)
    teach=data.target.double().numpy()
    teach/=teach.sum(1,keepdims=True)
    hs=huber_scale(raw['full_huber'],ts['old'].numpy(),mask,meta)
    ks=shrink(idx,s,teach,mask,raw['content_kl'],meta,weights(meta,idx))
    frozen_h=read(B/'reports/calibration_strength_report.json')['classes']
    scales=dict(full_huber=hs,full_kl=read(RUN/'initial/calibration.json')['classes'],
                content_huber={c:v['b'] for c,v in frozen_h.items()},content_kl=ks)
    h0,p0=quantities(s,mask)
    vals={}
    for arm in ARMS:
        coef=np.array([scales[arm][c] for c in cls])[:,None]
        h,p=quantities(s+coef*raw[arm],mask)
        vals[arm]=np.stack([h,p,h-h0,p-p0],axis=1)
    groups={'all':np.ones(len(idx),bool),**{c:cls==c for c in sorted(set(cls))},
            **{name:roles==name for name in ('bracket','following')}}
    result={}
    for name,take in groups.items():
        item=dict(positions=int(take.sum()),tasks=len(set(meta['goal'][take])),targets={},paired_contrasts={})
        for arm in ARMS:
            stats=cluster_mean(meta['goal'][take],vals[arm][take])
            item['targets'][arm]={metric:dict(mean=m,ci95=c) for metric,m,c in zip(
                ('entropy_nats','top1','delta_entropy_nats','delta_top1'),stats['mean'],stats['ci95'])}
        for key,(i,j) in PAIRINGS.items():
            pair=(vals[ARMS[i]]-vals[ARMS[j]])[take,:2];st=cluster_mean(meta['goal'][take],pair)
            item['paired_contrasts'][key]={metric:dict(mean=m,ci95=c) for metric,m,c in zip(('entropy_nats','top1'),st['mean'],st['ci95'])}
        for arm,former in [('full_kl','kl_projector'),('content_huber','huber_projector')]:
            for metric in ('entropy_nats','delta_top1'):
                a=item['targets'][arm][metric];b=prior['primary_stage_B_support'][name][metric][former]
                assert abs(a['mean']-b['mean'])<1e-6 and np.max(abs(np.array(a['ci95'])-b['ci95']))<1e-6
        result[name]=item
    save(DEST/'crossed_projectors.json',dict(groups=result,scales=scales,receipts=receipts,device='cpu',epochs=10,rank=8,
         positions=len(idx),tasks=512,task_bootstrap_draws=DRAWS,bootstrap_seed=SEED,fit_seed=FIT_SEED,
         calibration='Pooled ASIN-held-out predictions; Huber task-balanced slope clipped to [0,1]; KL task/category-balanced forward-KL minimum on [0,1].',
         raw_refit_differences=dict(full_huber_vs_B=diagnostics(raw['full_huber'],originals,mask),
                                    content_kl_vs_25_9b=diagnostics(raw['content_kl'],klprior,mask)),
         source_sha256=sources,prior_report_sha=digest_file(DEST/'tcal_entropy.json'),feature_digest=feature_hash.hexdigest()))


def write_report():
    report = read(DEST/'crossed_projectors.json')
    groups = report['groups']
    labels = ['原始 × Huber', '原始 × KL', '内容 × Huber', '内容 × KL']
    focus = ['bracket', 'following', 'format']
    order = [*focus, 'all', 'click', 'option', 'search', 'unknown']
    lines = [
        '# T_cal 交叉投影对照',
        '',
        '在首条命令的右括号处，固定标签时，两种 KL 投影的最高候选概率都低于对应的 Huber 投影。固定拟合方法时，原始标签的最高概率也都低于内容标签。右括号后的第一个位置则主要随拟合方法变化。下表给出同位置配对差及其不确定性。',
        '',
        '原始标签为中心化的完整包减无包，内容标签为中心化的完整包减同模板占位包。Huber 拟合中心化标签，前向 KL 拟合 softmax_C(s + 中心化标签) 的整行概率。生成投影目标时，把预测修正分别乘以类别系数 b 或 λ，加到同一学生 logits s 上，再取 softmax_C。两种方法都在原 ASIN 四折上产生题外预测。原始 × Huber 和内容 × KL 为本次 CPU 重拟合，另两列复用原冻结模型或折外预测。',
        '',
        '## 候选分布的熵（nats）',
    ]
    for metric, scale, digits in [('entropy_nats', 1, 4), ('delta_top1', 100, 3)]:
        if metric == 'delta_top1':
            lines.extend(['', '## 最高概率相对无包学生的变化（百分点）'])
        items = []
        for name in order:
            group = groups[name]
            items.append([name, group['positions'], *[
                estimate(group['targets'][arm][metric], scale, digits) for arm in ARMS]])
        table(lines, ['位置', '数量', *labels], items)
    lines.extend(['', '## 同一位置的直接比较', '',
                  '表中差值按列头顺序相减。方括号是差值本身的配对区间。'])
    comparisons = ['原始标签下 KL − Huber', '内容标签下 KL − Huber',
                   'Huber 下原始 − 内容', 'KL 下原始 − 内容']
    for metric, scale, digits, title in [('entropy_nats', 1, 4, '熵差（nats）'), ('top1', 100, 3, '最高概率差（百分点）')]:
        table(lines, [title, *comparisons], [
            [name, *[estimate(groups[name]['paired_contrasts'][k][metric], scale, digits)
                     for k in PAIRINGS]] for name in focus])
    lines.extend(['', '## 拟合与区间', '',
                  '四列使用阶段 B 同一批 34,825 个位置、512 个任务、同一无包学生表示和同一候选集 C。每个外折的 128 个任务及其 ASIN 均从该折拟合中排除。两个新组合均使用 rank-8 和原拟合种子，每折训练 10 个 epoch。Huber 的类别系数是原任务等权校准斜率截断至 [0, 1]；KL 的类别系数在 [0, 1] 内按原任务及类别权重最小化前向 KL。两者校准均汇总四折的留出预测。'])
    table(lines, ['训练类别系数', *labels], [[name, *[f"{report['scales'][arm][name]:.4f}" for arm in ARMS]]
          for name in sorted(report['scales']['full_huber'])])
    lines.extend(['', '统计时每个位置等权，按任务整簇重抽 10,000 次，报告 95% 百分位区间；重抽时固定投影器和收缩系数。bracket 是首条命令右括号所在 token，following 是下一 token，两行与 format 等词法类别可重叠。最高概率指每个分布在 C 上的最大概率，候选的身份可随分布改变。该指标描述候选分布的集中程度。',
                  '', '完整数据包含绝对 top-1、相对学生的熵差、各类别配对差、拟合记录及输入哈希。[机器可读结果](crossed_projectors.json) · [原熵表和无包学生参照](T_CAL_ENTROPY.md)'])
    (DEST/'CROSSED_PROJECTORS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
    write_report()
