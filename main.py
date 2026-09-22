"""
CLI entrypoint for NeuralNGEN.

Usage:
  python -u main.py train_multi --config configs/scoring_rule_multi_all.yml --gpu 0
  python -u main.py evaluate_multi --config configs/scoring_rule_multi_all.yml --gpu 0
"""

import argparse
from pathlib import Path

import yaml
import torch


def resolve_device(gpu_arg):
    if gpu_arg is None or gpu_arg == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{gpu_arg}"


def get_run_dir(cfg: dict) -> Path:
    """{run_dir_base}/{experiment_name}_{loss_type}/

    NOTE: if experiment_name in the config already ends in _{loss_type},
    this doubles the suffix -- keep experiment_name WITHOUT the loss_type
    suffix in the config; this function appends it.
    """
    experiment_name = cfg["experiment_name"]
    loss_type = cfg.get("loss_type", "both")
    run_dir_base = cfg.get("run_dir_base", "runs")
    return Path(run_dir_base) / f"{experiment_name}_{loss_type}"


class ScoringRuleConfig:
    def __init__(self, config_path: str, gpu_arg=None):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["device"] = resolve_device(gpu_arg)
        cfg["run_dir"] = str(get_run_dir(cfg))
        self.cfg = cfg

    def as_dict(self) -> dict:
        return self.cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train_multi", "evaluate_multi"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    config = ScoringRuleConfig(args.config, gpu_arg=args.gpu)
    cfg = config.as_dict()

    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Device: {cfg['device']}")
    print(f"Run dir: {cfg['run_dir']}")

    if args.mode == "train_multi":
        from src.trainig.train_multi_gauge import train
        train(cfg)
    elif args.mode == "evaluate_multi":
        from src.trainig.evaluate_multi_gauge import evaluate
        evaluate(cfg)


if __name__ == "__main__":
    main()
