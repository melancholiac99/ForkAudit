"""Line-delimited RPC process owning real community engines; CPU only."""
import argparse
import contextlib
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import re
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_value, write_once
from exploratory_shop_audit.community_smoke import plain


def parse_action_fixed(action):
    if not isinstance(action, str):
        return action, None
    match = re.fullmatch(r"([A-Za-z_]\w*)\[(.*)\]", action, flags=re.DOTALL)
    return match.groups() if match else (action, None)


class Engine:
    def __init__(self, root, products, index, snapshots):
        os.environ["SHOP_SEARCH_INDEX"] = str(index.resolve())
        sys.path.insert(0, str(root / "src"))
        self.shop = root / "environments/ShopSimulator/shop_env"
        sys.path.insert(0, str(self.shop))
        import web_agent_site.engine.engine as core
        import web_agent_site.envs.web_agent_text_env as module
        from web_agent_site.engine.reward import evaluate_purchase
        from web_agent_site.engine.variant_price import resolve_variant_price
        from shopping_grpo.environment.observation import render_structured_observation
        core.parse_action = parse_action_fixed
        module.parse_action = parse_action_fixed
        self.env_class = module.WebAgentTextEnv
        self.evaluate = evaluate_purchase
        self.resolve_price = resolve_variant_price
        self.render = render_structured_observation
        self.products_path = str(products)
        self.snapshots = snapshots
        snapshots.mkdir(parents=True, exist_ok=True)
        price_path, goal_path = snapshots / "product_prices_frozen.json", snapshots / "goals_frozen.json"
        if price_path.exists() != goal_path.exists():
            raise RuntimeError("Incomplete frozen table pair")
        if price_path.exists():
            prices = json.loads(price_path.read_text())
            goals = json.loads(goal_path.read_text())
            loader = module.load_products

            def verified_loader(*args, **kwargs):
                all_products, lookup, generated_prices, attrs = loader(*args, **kwargs)
                if generated_prices != prices:
                    raise RuntimeError("Native deterministic price table changed")
                return all_products, lookup, deepcopy(prices), attrs

            module.load_products = verified_loader
            module.get_goals = lambda *a, **kw: deepcopy(goals)
        self.base = self.env_class(observation_mode="text", file_path=self.products_path,
                                  if_persona=False, split="train", shuffle_goals=False, shift_goals=False)
        self.server = self.base.server
        write_once(price_path, self.server.product_prices)
        write_once(goal_path, self.server.goals)
        self.hashes = {"prices": digest_value(self.server.product_prices), "goals": digest_value(self.server.goals)}
        self.envs = {}
        self.prefixes = {}

    def open(self, key, goal_index, prefix=()):
        if key in self.envs:
            self.close(key)
        goal = self.server.goals[goal_index]
        if self.server.product_item_dict[goal["asin"]]["tag"] != "train":
            raise RuntimeError("Refuse policy or replay access to eval task")
        env = self.env_class(observation_mode="text", server=self.server,
                             file_path=self.products_path, if_persona=False,
                             split="train", shuffle_goals=False, shift_goals=False)
        self.server.user_sessions.pop(key, None)
        env.reset(idx=goal_index, session=key)
        self.envs[key] = env
        self.prefixes[key] = []
        for action in prefix:
            result = self.step(key, action)
            if result["status"].get("done"):
                raise RuntimeError("A restored decision prefix contains a terminal action")
        return self.state(key)

    def close(self, key):
        self.envs.pop(key, None)
        self.prefixes.pop(key, None)
        self.server.user_sessions.pop(key, None)

    def score(self, product, goal, options, resolution):
        before = digest_value(product)
        result = self.evaluate(deepcopy(product), deepcopy(goal), selected_options=deepcopy(options),
                               price_resolution=deepcopy(resolution), rewards=self.server.environment_config["reward"])
        diagnostic_goal = deepcopy(goal)
        diagnostic_goal["price_upper"] = None
        diagnostic = self.evaluate(deepcopy(product), diagnostic_goal, selected_options=deepcopy(options),
                                   price_resolution=deepcopy(resolution), rewards=self.server.environment_config["reward"])
        if digest_value(product) != before:
            raise RuntimeError("Offline scoring mutated product")
        return {"reward": result.reward, "no_budget_gate": diagnostic.reward,
                "valid": result.reward_valid, "detail": result.to_dict(),
                "diagnostic_detail": diagnostic.to_dict()}

    def state(self, key):
        env = self.envs[key]
        session = self.server.user_sessions[key]
        public = env.structured_observation()
        available = env.get_available_actions()
        terminal = bool(session["done"])
        stop = not terminal and "buy now" in available["clickables"]
        product = self.server.product_item_dict.get(session.get("asin"))
        score = self.score(product, session["goal"], session["options"], session.get("price_resolution")) if stop else None
        return {"observation": self.render(public), "native_observation": env.observation,
                "public": public, "available": available, "done": terminal,
                "remaining": self.server.max_steps - len(self.prefixes[key]),
                "stop_available": stop, "stop_score": score,
                "backend": {"session": plain(session), "current_product": deepcopy(product)},
                "task": session["goal"]["instruction_text"], "tables": self.hashes}

    def step(self, key, action):
        env = self.envs[key]
        if self.server.user_sessions[key]["done"]:
            raise RuntimeError("Step after terminal")
        before = self.state(key)
        parsed_name, parsed_arg = parse_action_fixed(action)
        normalized = parsed_arg.lower() if isinstance(parsed_arg, str) else parsed_arg
        legal = ((parsed_name == "search" and bool(normalized)) or parsed_name == "finish" or
                 (parsed_name == "click" and normalized in before["available"]["clickables"] and normalized != "search"))
        product = self.server.product_item_dict.get(self.server.user_sessions[key].get("asin"))
        before_hash = digest_value(product) if product else None
        obs, status, info = env.step(action)
        self.prefixes[key].append(action)
        if before_hash is not None and digest_value(product) != before_hash:
            raise RuntimeError("Actual environment action mutated shared product")
        after = self.state(key)
        diagnostic = None
        if status.get("done"):
            if status.get("purchase"):
                session = self.server.user_sessions[key]
                scored = self.score(self.server.product_item_dict[session["asin"]], session["goal"],
                                    session["options"], session.get("price_resolution"))
                if scored["reward"] != status["reward"]:
                    raise RuntimeError("Actual buy versus offline score mismatch")
                diagnostic = scored["no_budget_gate"]
            else:
                diagnostic = status["reward"]
        return {"state": after, "status": plain(status), "legal": bool(legal),
                "no_budget_gate": diagnostic, "info": plain(info)}

    def parser_audit(self):
        count = 0
        for product in self.server.all_products:
            for values in product["options"].values():
                for value in values:
                    if parse_action_fixed(f"click[{value}]") != ("click", value):
                        raise RuntimeError("Parser failed a native normalized option")
                    count += 1
        return {"option_values": count, "failures": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--products", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--snapshots", type=Path, required=True)
    args = p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine(args.root.resolve(), args.products.resolve(), args.index.resolve(), args.snapshots.resolve())
    print(json.dumps({"ready": True, "tables": engine.hashes}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                cmd = request["cmd"]
                if cmd == "quit":
                    return
                if cmd == "open": result = engine.open(request["key"], request["goal_index"], request.get("prefix", []))
                elif cmd == "state": result = engine.state(request["key"])
                elif cmd == "step": result = engine.step(request["key"], request["action"])
                elif cmd == "close": result = engine.close(request["key"])
                elif cmd == "parser_audit": result = engine.parser_audit()
                else: raise ValueError(cmd)
            response = {"ok": True, "result": result}
        except Exception:
            response = {"ok": False, "error": traceback.format_exc()}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
