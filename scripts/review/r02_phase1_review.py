"""Review part 2: A_replay sign/values, chosen-not-admissible, provisional M5 logprob shift.
PROVISIONAL conf = mean logprob of the action-span tokens of the stored rollout response, i.e. conditioned on the
realized <think> (one candidate reading of D-003, still OPEN). Not the frozen arm-A definition."""
import collections, glob, json, os
import numpy as np, pandas as pd
from transformers import AutoTokenizer
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_parquet(f"{R}/data/decision_points.parquet")
roll = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{R}/data/rollout/s*.json")}
D = df[df.cf_defined]
print("A_replay value counts (top):", D.A_replay.round(3).value_counts().head(8).to_dict())
print("sign among pivotal: pos", int((D.A_replay > 0).sum()), "neg", int((D.A_replay < 0).sum()))
adm_ok = [x.chosen_action in set(x.admissible_actions) for _, x in df.iterrows()]
df["chosen_adm"] = adm_ok
print("chosen not admissible: all", int((~df.chosen_adm).sum()), " in D", int((~df.chosen_adm[df.cf_defined]).sum()),
      " excluded", int((~df.chosen_adm[~df.cf_defined]).sum()))

tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
conf, nspan = [], 0
for _, x in df.iterrows():
    s = roll[x.traj_id]["steps"][x.turn - 1]
    ids, lps, text = s["response_token_ids"], s["response_logprobs"], s["response"]
    a = s["action"]; k = text.rfind("</think>"); low = text.lower()
    j = low.find(a, k if k >= 0 else 0)
    if j < 0 or not lps:
        conf.append(np.nan); continue
    # token char offsets by incremental decode
    ends, acc = [], ""
    for t in range(len(ids)):
        ends.append(len(tok.decode(ids[: t + 1])))
    starts = [0] + ends[:-1]
    sel = [t for t in range(len(ids)) if ends[t] > j and starts[t] < j + len(a)]
    conf.append(float(np.mean([lps[t] for t in sel]))); nspan += 1
df["conf_prov"] = conf
inc, exc = df[df.cf_defined].conf_prov.dropna(), df[~df.cf_defined].conf_prov.dropna()
rng = np.random.default_rng(0)
bs = [rng.choice(inc.values, len(inc)).mean() - rng.choice(exc.values, len(exc)).mean() for _ in range(10000)]
print(f"provisional conf found for {nspan}/{len(df)}; included mean {inc.mean():.3f} (n={len(inc)}), excluded {exc.mean():.3f} (n={len(exc)})")
print(f"included-minus-excluded {inc.mean()-exc.mean():+.3f}  95% boot CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]   [CWGT -0.70 [-0.81,-0.59]]")
print("excluded conf quantiles", np.round(np.percentile(exc, [10, 50, 90]), 4), " included", np.round(np.percentile(inc, [10, 50, 90]), 4))
