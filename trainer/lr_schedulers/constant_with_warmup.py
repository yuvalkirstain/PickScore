from dataclasses import dataclass

from omegaconf import II
from transformers import get_constant_schedule_with_warmup


@dataclass
class ConstantWithWarmupLRSchedulerConfig:
    _target_: str = "trainer.lr_schedulers.constant_with_warmup.instantiate_dummy_lr_scheduler"
    lr: float = II("optimizer.lr")
    lr_warmup_steps: int = 500
    total_num_steps: int = II("accelerator.max_steps")


def instantiate_dummy_lr_scheduler(cfg: ConstantWithWarmupLRSchedulerConfig, optimizer):
    return get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
    )
