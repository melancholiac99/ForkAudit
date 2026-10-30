from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'e1.json').read_text())
lines=['P4 E0/E1 实验报告（2026-10-30）','',
'结论：E1 未支持预期的完整分层模式，且世界信号正对照失败。依预注册条款，不能将内省未过线解释为已证实“看不见分叉”；当前 E1 读法为 R-MIXED，不启动 E3。E2 尚未运行，P4-5/6/7 未判定，完整 E1+E2 表序裁决待补。',
'', '完成：E0 分层重算及逐点核对；预注册提交 20f0976606ca01929bce67d3b850df1045fe8c4e；E1 全部信号、三层、10,000 次轨迹 bootstrap、1,000 次层内×轨迹×进度三分位 shuffle；独立库1+2动作收益与Gold第二读出；标签量。',
'查新：目标式原文核对暂未发现精确停止条件匹配；检索范围、近邻工作与未完整获取的原文记录于 novelty_check.json，不声称穷尽文献。',
'资源：无可见GPU，容器2 GiB内存、0.5 CPU。用户指示“先做能做的”。E2输入已准备873点×4条件=3492个上下文，每个模型17460个候选序列；7B/32B均未打分。',
'', '预测对账：',
'P4-1：按字面判据成立（全部六项均未超过I层shuffle上分位），但正对照失败使该负结果不可作盲区证据。',
'P4-2：不成立，C_sel与B_multi的U−I点估计均为负，CI含0。',
'P4-3：不成立，I层策略G_u显著为负，U层不显著为正。',
'P4-4：两个点估计方向成立；符号正比例的层间差显著，Live上chosen份额的差CI含0。',
'正对照：失败，I层W1pure_multi AUROC 0.5962，未超过shuffle上分位0.6120。',
'', '状态级：AUROC [95% CI]；shuffle 97.5%分位；按预注册方向，不翻转。']
for layer,group in r['state'].items():
 lines.append('\n'+layer)
 for name,z in group.items():lines.append(f"  {name}: {z['auroc']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}]; shuffle={z['shuffle_ci95'][1]:.4f}; exceeds={z['above_shuffle']}")
lines.append('\n层间 U−I AUROC：')
for name,z in r['U_minus_I'].items():lines.append(f"  {name}: {z['diff']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}]")
lines.append('\n动作选择：只在各层Fork上；百分点；J(p,q)是固定五候选上的离线重加权收益，并非新策略在线成功率。')
for layer,srcs in r['action'].items():
 for src,models in srcs.items():
  for model,metrics in models.items():
   for metric,z in metrics.items():lines.append(f"  {layer}/{src}/{model}/{metric}: {100*z['estimate']:+.2f} [{100*z['ci95'][0]:+.2f}, {100*z['ci95'][1]:+.2f}], n={z['n']}, trajectories={z['n_traj']}")
lines.append('\n标签量：比例，95%CI；chosen share为库3–4×5–6的交叉乘积总量比。')
for layer,metrics in r['labels'].items():
 for name,z in metrics.items():lines.append(f"  {layer}/{name}: {z['estimate']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}], n={z['n']}")
for name,z in r['label_U_minus_I'].items():lines.append(f"  U−I/{name}: {z['diff']:.4f} [{z['ci95'][0]:.4f}, {z['ci95'][1]:.4f}]")
lines+=['','解释边界：',
'1. I层B_multi/C_sel AUROC约0.68，但其层内轨迹进度shuffle基线也很高；不能把“未超过该基线”写成“AUROC只有随机水平”。',
'2. target_revealed在I、U层恒为1，在N层恒为0，层内AUROC必为0.5。分层条件固定了该强世界特征；多特征正对照失败可能反映条件化后剩余变化减少，不能单凭此说世界信息无用。该解释是结果后的诊断，不更改判据。',
'3. 目标反馈选臂在I层效果很强，Gold读出同向；这是动作选择量。它不替代状态级W1正对照，也不证明32B评判者能利用该反馈。',
'4. I层策略分布负收益在Gold上仍为负；U层主读出CI含0，Gold上为负，未支持“利用受限层能选好动作”。裸命令打分不是含思考的自由生成边际分布。',
'5. 模型变强后内省信用更差尚无实验依据；E2/E3未运行。非显著也不是等效性证明。',
'','复现：',
'  /root/miniconda3/envs/m0/bin/python exploratory_P4/p4_strata.py',
'  OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/m0/bin/python exploratory_P4/p4_e1_introspection.py',
'  OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/m0/bin/python exploratory_P4/verify_cpu.py',
'  /root/miniconda3/envs/m0/bin/python exploratory_P4/render_results.py',
'','GPU后续入口（尚未经过真实GPU验证）：',
'  p4_e2_judge_score.py --model PATH --name 7b --revision PINNED_REVISION',
'  p4_e2_judge_score.py --model PATH --name 32b --revision PINNED_REVISION',
'  p4_e2_analyze.py （仅两个模型全量完成且数值校验通过后产出e2.json）',
'32B revision必须在运行前写DECISIONS，模型须完整本地下载；共享前缀/直接打分TV阈值0.005，每模型先核验。',
'','独立核验：见cpu_verification.json。逐点AUROC对照sklearn，40次加权bootstrap逐项对照，动作收益与既有deep独立提取核对，10,656个输入SHA256及3,492个E2请求核验通过。']
(OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for k,layer in enumerate(['I','U','N']):
 for j,model in enumerate(['policy','world']):
  z=r['action'][layer]['12'][model]['G_u'];y=k+(-.12 if j==0 else .12);v=100*z['estimate'];lo,hi=np.array(z['ci95'])*100
  axs[0].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=['#4056a1','#d46a28'][j],label=model if k==0 else None,capsize=3)
axs[0].set(yticks=range(3),yticklabels=['I (184 Fork)','U (202 Fork)','N (65 Fork)'],xlabel='Gain over uniform (percentage points)',title='Action selection: independent banks 1+2');axs[0].axvline(0,color='gray',lw=1);axs[0].legend()
for k,name in enumerate(['C_sel','B_multi','W1pure_multi']):
 z=r['U_minus_I'][name];v=z['diff'];lo,hi=z['ci95'];axs[1].errorbar(v,k,xerr=[[v-lo],[hi-v]],fmt='o',color='#4056a1',capsize=3)
axs[1].set(yticks=range(3),yticklabels=['C_sel','B_multi','W1pure_multi'],xlabel='AUROC(U) − AUROC(I)',title='State discrimination: paired cluster CI');axs[1].axvline(0,color='gray',lw=1)
fig.suptitle('P4 E1: proposed pattern not supported; positive control failed')
fig.savefig(OUT/'e1_summary.png',dpi=180);fig.savefig(OUT/'e1_summary.svg');plt.close(fig)
status=dict(E0='completed (targeted novelty review limitations recorded)',E1='completed and independently verified',E1_reading='R-MIXED: positive control failed',E2='prepared, not scored: no GPU',E3='not started; preregistered gate not satisfied',paper_mainline='not changed',user_instruction='先做能做的')
(OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
