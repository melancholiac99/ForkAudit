"""Train-only native feasibility audit; candidates are emitted by the public page."""
from pathlib import Path
from collections import Counter
import json,time
from .b_collect import RUN,PARENT
from exploratory_shop_audit.grpo_forced_audit.interface import Engine
from exploratory_shop_audit.grpo_forced_audit.cpu import complete_configurations
from exploratory_shop_audit.grpo_forced_audit.common import save,read

def build():
    engine=Engine(RUN)
    allowed={r['goal_index'] for r in read(PARENT/'asin_split_manifest.json')['T_cal']}
    cache={};audit=[];counts=Counter()
    for offset in [0,*range(8,512,64)]:
        p=RUN/f'tape_{offset:03d}.jsonl'
        while not p.exists():time.sleep(5)
        for line in p.open():
            row=json.loads(line);g=row['goal']
            assert g in allowed
            if row['page']!='search_results':continue
            counts['result_states']+=1
            goal=engine.server.goals[g];gold=goal['asin']
            visible=[a for a in row['legal_clicks'] if a in row['products']]
            if gold not in visible:counts['gold_absent']+=1;continue
            good=[]
            for asin in visible:
                if (g,asin) not in cache:
                    prod=engine.server.product_item_dict[asin];okay=False
                    for options in complete_configurations(prod):
                        res=engine.evaluate(prod,goal,selected_options=options,price_resolution=None,rewards=engine.server.environment_config['reward'])
                        counts['configurations_checked']+=1
                        if res.reward_valid and res.reward_type in ('gold_purchase','valid_alternative_purchase'):okay=True
                    cache[g,asin]=okay
                if cache[g,asin]:good.append(asin)
            if gold not in good:counts['gold_infeasible']+=1;continue
            alt=[a for a in good if a!=gold]
            if not alt:counts['no_equivalent_alternative']+=1;continue
            counts['eligible_states']+=1
            audit.append(dict(goal=g,step=row['step'],gold=gold,alternatives=alt))
    save(RUN/'equivalent_pool.json',dict(counts=counts,states=audit,scope='T_cal only; exhaustive native option evaluation after visible nomination'))
if __name__=='__main__':build()
