#!/usr/bin/env python3
"""Preflight for running the CTHA single-GPU trainer on the original server."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--require-gpu", action="store_true")
    a = p.parse_args()
    out = dict(model=str(a.model.resolve()), model_exists=a.model.is_dir(), packages={}, cuda=None,
               alfworld_game_exists=None)
    for name in ("numpy", "torch", "transformers", "peft", "vllm", "alfworld", "textworld"):
        try:
            mod = importlib.import_module(name)
            out["packages"][name] = getattr(mod, "__version__", "installed")
        except Exception as e:
            out["packages"][name] = f"ERROR: {type(e).__name__}: {e}"
    if not str(out["packages"]["torch"]).startswith("ERROR"):
        import torch
        out["cuda"] = dict(available=torch.cuda.is_available(),
                           device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import alfenv
        games = json.loads((ROOT / "exploratory_E4" / "games.json").read_text())["train"]
        path = Path(alfenv.ALFWORLD_DATA) / games[0]["game_file"]
        out["alfworld_game_exists"] = path.is_file()
        out["alfworld_sample_path"] = str(path)
    except Exception as e:
        out["alfworld_game_exists"] = False
        out["alfworld_error"] = str(e)
    out["ready"] = (out["model_exists"] and out["alfworld_game_exists"]
                    and all(not str(v).startswith("ERROR") for v in out["packages"].values())
                    and (not a.require_gpu or bool(out["cuda"] and out["cuda"]["available"])))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if not out["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
