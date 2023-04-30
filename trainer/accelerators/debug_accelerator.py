from dataclasses import dataclass
from accelerate import Accelerator
from trainer.accelerators.base_accelerator import BaseAcceleratorConfig, BaseAccelerator


@dataclass
class DebugAcceleratorConfig(BaseAcceleratorConfig):
    _target_: str = "trainer.accelerators.debug_accelerator.DebugAccelerator"


class DebugAccelerator(BaseAccelerator):
    def __init__(self, cfg: DebugAcceleratorConfig):
        super().__init__(cfg)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=cfg.log_with,
            project_dir=cfg.output_dir,
            dynamo_backend=cfg.dynamo_backend,
        )
        self.post_init()
