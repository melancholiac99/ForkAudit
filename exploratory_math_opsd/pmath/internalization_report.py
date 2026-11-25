"""Build final tables and input provenance after all eight state reports exist."""
import json
from internalization import EVIDENCE, GROUPS, STATES, WORK
from launch import sha, save

def main():
    reports = {s:json.loads((EVIDENCE/f'{s}.json').read_text()) for s in STATES}
    def interval(state,key):
        m=reports[state]['metrics'][key]
        return f"{m['value']:.6f} [{m['ci95'][0]:.6f}, {m['ci95'][1]:.6f}]"
    lines=['# 固定探针的内化度量','','1,500 题，1,185,192 个回答位置，温度 1.1。每题仅使用留出它的投影器。区间按题整簇重抽 10,000 次（20261124）；KL 为 nats/位置。','',
           '| 状态 | G | U | 题外斜率 | 截后 λ |','|---|---:|---:|---:|---:|']
    for s in STATES:
        lines.append('| '+' | '.join([s]+[interval(s,k) for k in ('G','U','raw_slope','lambda')])+' |')
    for key in ('G','U'):
        lines+=['',f'## {key} 的有符号词表组贡献','','| 状态 | '+' | '.join(GROUPS)+' |','|---|'+'---:|'*len(GROUPS)]
        for s in STATES:
            lines.append('| '+' | '.join([s]+[interval(s,key+'/'+g) for g in GROUPS])+' |')
    (EVIDENCE/'README.md').write_text('\n'.join(lines)+'\n')
    paths=[WORK/'head.pt']+[WORK/f'rollouts-{k}.pt' for k in range(4)]
    save(EVIDENCE/'inputs.json',dict(archives={p.name:sha(p) for p in paths},
                                    adapters={s:r['adapter_sha256'] for s,r in reports.items()}))

if __name__=='__main__':main()
