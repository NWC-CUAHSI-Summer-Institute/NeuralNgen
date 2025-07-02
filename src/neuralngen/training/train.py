import argparse
from pathlib import Path

from neuralngen.models.ngenlstm import NgenLSTM
from neuralngen.dataset import BaseDataset
from neuralngen.training.basetrainer import BaseTrainer
from neuralngen.utils import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load config from YAML file
    cfg = Config(Path(args.config))

    model = NgenLSTM(cfg)
    trainer = BaseTrainer(cfg, model, BaseDataset)
    trainer.train()

if __name__ == "__main__":
    main()