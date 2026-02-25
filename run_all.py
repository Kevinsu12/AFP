"""Run pre-training followed by backtest in a single command.

Usage:
    python run_all.py
"""

from rl.configs.sac_config import Config
from rl.cli.pretrain import run_pretrain
from rl.cli.backtest import main as run_backtest


if __name__ == "__main__":
    cfg = Config.default()

    print("\n" + "=" * 70)
    print("  STEP 1 / 2 : PRE-TRAINING")
    print("=" * 70 + "\n")
    run_pretrain(cfg)

    print("\n" + "=" * 70)
    print("  STEP 2 / 2 : BACKTEST")
    print("=" * 70 + "\n")
    run_backtest(cfg)
