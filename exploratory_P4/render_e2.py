"""Render P4 completion report from verified E1, X and E2 artifacts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent

def main():
 r=json.loads((OUT/'e2.json').read_text());verify=json.loads((OUT/'e2_verification.json').read_text());assert verify['passed']
 e1=json.loads((OUT/'e1.json').read_text());x=json.loads((OUT/'x.json').read_text())
 def fmt(v):
  if v['estimate'] is None:return 'undefined (n=0)'
  lo,hi=v['ci95'];return f"{100*v['estimate']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] pp"+(f"; n={v['n']}, trajectories={v['n_traj']}" if 'n' in v else '')
 lines=['P4 E2 完成报告','',f"P4最终读法（沿用已登记的正对照失败条款）：{r['verdict']}。E3_allowed={r['E3_allowed']}。",'E1正对照失败，不把内省未过线当作盲区证明；E2逐项结果如下，不能用E2替代失败的状态级正对照。',
 'P4-X独立探索性读法：'+x['reading']+'；不改P4原预测。',
 '', '设计：873个I/U层Live点，每模型3492上下文/17460候选序列；主分析386个Fork（I=184,U=202）。',
 '模型：Qwen2.5-7B-Instruct与Qwen2.5-32B-Instruct，固定版本；BF16 transformer、FP32词表投影；固定<action>回答前缀，无实现的思考文本。',
 'S=决策时上下文；L=加完整factual日志与终局奖励；F=加带动作对应的一步反馈；Fx=固定置换替代动作反馈。',
 '主读出库1+2（6次/臂），Gold为独立20次/臂第二读出。10,000次轨迹bootstrap，同一Live轨迹集与seed 30102026。',
 '这些是固定候选集上的离线重加权成功率，不是新策略的在线评估或训练增益。',
 '', '预测对账：']
 for name,val in e1['predictions'].items():lines.append('  E1 '+name+': '+str(val))
 for name,val in r['predictions'].items():lines.append('  E2 '+name+': '+json.dumps(val,ensure_ascii=False))
 lines+=['  P4-8/9/10: 未运行，E3启动条件不满足。' if not r['E3_allowed'] else '  E3条件满足，需另行启动。',
 '  P4-4_direction是点估计方向；chosen份额层间差CI含0，不能写作两项均显著。',
 '', 'E2条件收益表（百分点，95%CI）：']
 for src in ['12','gold']:
  lines.append('\nOutcome source: '+src)
  for layer in ['I','U']:
   for model in ['7b','32b']:
    for condition in ['S','L','F','Fx']:
     for mode in ['soft','argmax']:
      for metric in ['G_u','G_c']:
       key='/'.join([model,condition,mode,metric,layer]);lines.append('  '+key+': '+fmt(r['table'][src][key]))
  lines.append('Paired contrasts:')
  for name,val in r['contrasts'][src].items():lines.append('  '+name+': '+fmt(val))
 lines+=['','数值校验及完整性：',json.dumps(verify,indent=2,ensure_ascii=False),
 '', '判定解释：当前分析器在E1世界正对照失败时优先返回R-MIXED（P4 §3 E1明确条款）。§4表序中的R-NULL/R-PRIOR条件另列供审阅，未用其更改登记的正对照失败处理。']
 c=r['contrasts']['12'];zero=lambda a:a['ci95'] is not None and a['ci95'][0]<=0<=a['ci95'][1]
 null=not any(r['predictions']['P4-6'].values());prior=zero(c['P4-5_interaction']) and r['table']['12']['32b/S/soft/G_u/I']['ci95'][0]>0
 lines += [f'  表序单项条件：R-NULL={null}; R-PRIOR={prior}。',
 '', '限制：I/U是目标实例文本代理；P_E或层内非显著不意味着等效；F/Fx检验动作反馈对应关系，不是纯粹因果中介检验；单对7B/32B不能外推普遍规模律。',
 '数值TV检查只覆盖固定样例，不是对全部上下文的统一误差保证。',
 '', '复现入口：',
 '  p4_e2_judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --name 7b --revision a09a35458c702b33eeacc393d103063234e8bc28',
 '  p4_e2_judge_score.py --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct --name 32b --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
 '  p4_e2_analyze.py; verify_e2.py; render_e2.py',
 '所有命令使用 /root/miniconda3/envs/m0/bin/python，在仓库根目录运行。',
 '权重文件不上传GitHub；精确版本、文件哈希和下载来源随结果记录。']
 (OUT/'E2_REPORT.txt').write_text('\n'.join(lines)+'\n')
 fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
 for ax,layer in zip(axs,['I','U']):
  for model,offset,color in [('7b',-.12,'#4056a1'),('32b',.12,'#d46a28')]:
   for i,condition in enumerate(['S','L','F','Fx']):
    v=r['table']['12'][f'{model}/{condition}/soft/G_u/{layer}'];m=100*v['estimate'];lo,hi=100*np.array(v['ci95'])
    ax.errorbar(i+offset,m,yerr=[[m-lo],[hi-m]],fmt='o',color=color,capsize=3,label=model if i==0 else None)
  ax.set(xticks=range(4),xticklabels=['S: context','L: full log','F: feedback','Fx: permuted'],ylabel='Gain over uniform (percentage points)',title=f'{layer} stratum: Fork points');ax.axhline(0,color='gray',lw=1);ax.legend()
 fig.suptitle('P4 E2: fixed candidate selection, independent banks 1+2')
 fig.savefig(OUT/'e2_summary.png',dpi=180);fig.savefig(OUT/'e2_summary.svg');plt.close(fig)
 status=dict(E0='completed targeted review; scope limitations in novelty_check.json',E1='completed and verified',E1_reading='R-MIXED: positive control failed',P4_X=x['reading'],E2='completed both models, analyzed and independently verified',P4_reading=r['verdict'],E3='not triggered by preregistered gate' if not r['E3_allowed'] else 'pending required execution',paper_mainline='not automatically changed')
 (OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(status,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
