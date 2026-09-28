"""PSA-MIL: probabilistic spatial attention for whole-slide classification."""

from psa_mil.models.psa import PSAMIL

__all__ = ["PSAMIL", "train_from_config"]


def train_from_config(config_path: str):
    """Train PSA-MIL from a YAML config. Imported lazily so the model can be used without Lightning."""
    from psa_mil.training.loop import train_from_config as _train

    return _train(config_path)
