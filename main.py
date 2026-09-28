"""Backward-compatible entry point. Prefer ``python train.py --config ...``."""

from psa_mil.training.loop import main

if __name__ == "__main__":
    main()
