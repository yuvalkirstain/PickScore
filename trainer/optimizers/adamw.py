from dataclasses import dataclass


@dataclass
class AdamWOptimizerConfig:
    _target_: str = "torch.optim.adamw.AdamW"
    lr: float = 1e-6

