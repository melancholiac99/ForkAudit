"""E4 step 0: freeze the game lists (run once, commit games.json before any E4 rollout).

train: 400 ALFWorld train games, stratified by task type, excluding M0's 100 tasks (data/task_list.json).
tune : 40 further train games (learning-rate selection only, PREREG_E4 §5).
eval : all valid_seen (eval_in_distribution) and valid_unseen (eval_out_of_distribution) games.
"""
import json
import os

import e4_common as E

import alfenv  # noqa: E402  (server only; sets ALFWORLD_DATA)


def main():
    assert not os.path.exists(E.GAMES_PATH), "games.json already frozen"
    train = alfenv.collect_games("train")
    m0 = json.load(open(os.path.join(E.ROOT, "data", "task_list.json")))["tasks"]
    used = {os.path.join(alfenv.ALFWORLD_DATA, t["game_file"]) for t in m0}
    assert len(used) == 100
    tr = E.stratified_draw(train, E.N_TRAIN, used, E.QUOTA_SEED)
    tu = E.stratified_draw(train, E.N_TUNE, used | {g for g, _ in tr}, E.QUOTA_SEED + 1)
    rel = lambda g: os.path.relpath(g, alfenv.ALFWORLD_DATA)

    def entries(lst, first):
        return [dict(seed=first + i, game_file=rel(g), task_type=t, game_sha256=E.file_sha256(g)) for i, (g, t) in enumerate(lst)]

    seen = alfenv.collect_games("eval_in_distribution")
    unseen = alfenv.collect_games("eval_out_of_distribution")
    out = dict(train=entries(tr, E.SEED_BASE), tune=entries(tu, E.SEED_BASE + E.N_TRAIN),
               valid_seen=entries(seen, 2000), valid_unseen=entries(unseen, 3000),
               rule="stratified_draw(train minus M0 task_list, quotas by sorted task type, seed QUOTA_SEED / +1)",
               quota_seed=E.QUOTA_SEED, alfworld_data=alfenv.ALFWORLD_DATA)
    assert len({e["game_file"] for k in ("train", "tune") for e in out[k]} & {rel(g) for g in used}) == 0
    E.write_json(E.GAMES_PATH, out)
    print({k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
