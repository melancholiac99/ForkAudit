"""Apply the native purchase rule to every product on an archived page."""
import sys
from pathlib import Path
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, read,save


def qualify(anchor_folder, pool, goals, result):
    from exploratory_shop_audit.grpo_forced_audit.interface import Engine
    from exploratory_shop_audit.derive_environment import configurations
    PARENT=ROOT/'.runtime/runs/shop_visible_opd_v04_s16_20261120'
    engine=Engine(PARENT)
    good={}
    for g in goals:
        state=read(anchor_folder/f'{pool}_{g}.json')['state']
        if state is None:
            good[str(g)]=None
            continue
        hits=[]
        for prod in state['public'].get('products',[]):
            item=engine.server.product_item_dict[prod['asin']]
            for opt in configurations(item):
                v=engine.evaluate(item,engine.server.goals[g],selected_options=opt,price_resolution=None,
                                  rewards=engine.server.environment_config['reward'])
                if v.reward_valid and v.reward_type in ('gold_purchase','valid_alternative_purchase'):
                    hits.append(prod['asin'].casefold())
                    break
        good[str(g)]=hits
    save(result,good)


if __name__=='__main__':
    src=Path(sys.argv[1])
    qualify(src,sys.argv[2],[int(g) for g in sys.argv[3:]],src/'qualify.json')
