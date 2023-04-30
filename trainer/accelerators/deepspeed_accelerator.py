import os
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate.utils import PrecisionType
from accelerate import Accelerator, DeepSpeedPlugin
from omegaconf import OmegaConf, MISSING, II

from trainer.accelerators.base_accelerator import BaseAcceleratorConfig, BaseAccelerator


@dataclass
class MixedPrecisionConfig:
    enabled: bool = MISSING


@dataclass
class DeepSpeedConfig:
    fp16: MixedPrecisionConfig = MixedPrecisionConfig(enabled=False)
    bf16: MixedPrecisionConfig = MixedPrecisionConfig(enabled=False)
    optimizer: dict = field(default_factory=lambda: {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
            "adam_w_mode": True
        }
    })
    scheduler: dict = field(default_factory=lambda: {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    })
    zero_optimization: dict = field(default_factory=lambda: {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True
    })
    gradient_accumulation_steps: int = 16
    gradient_clipping: float = 1.0
    steps_per_print: int = 1
    train_batch_size: str = "auto"
    train_micro_batch_size_per_gpu: str = "auto"
    #     train_micro_batch_size_per_gpu: int = II("dataset.batch_size")
    wall_clock_breakdown: bool = False


@dataclass
class DeepSpeedAcceleratorConfig(BaseAcceleratorConfig):
    _target_: str = "trainer.accelerators.deepspeed_accelerator.DeepSpeedAccelerator"
    deepspeed: DeepSpeedConfig = DeepSpeedConfig()
    deepspeed_final: Any = None


class DeepSpeedAccelerator(BaseAccelerator):
    def __init__(self, cfg: DeepSpeedAcceleratorConfig):
        super().__init__(cfg)
        self.set_mixed_precision()
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(self.cfg.deepspeed, resolve=True),
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
        )
        self.cfg.deepspeed_final = OmegaConf.create(deepspeed_plugin.deepspeed_config)
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.log_with,
            project_dir=self.cfg.output_dir,
            dynamo_backend=self.cfg.dynamo_backend,
        )
        self.post_init()

    def set_mixed_precision(self):
        if self.cfg.mixed_precision == PrecisionType.BF16:
            self.cfg.deepspeed.bf16.enabled = True
            self.cfg.deepspeed.fp16.enabled = False
        elif self.cfg.mixed_precision == PrecisionType.FP16:
            self.cfg.deepspeed.fp16.enabled = True
            self.cfg.deepspeed.bf16.enabled = False
        else:
            self.cfg.deepspeed.fp16.enabled = False
            self.cfg.deepspeed.bf16.enabled = False

    def prepare(self, *args, device_placement=None):
        prepared = self.accelerator.prepare(*args, device_placement=device_placement)
        for obj in prepared:
            if isinstance(obj, torch.nn.Module):
                if self.cfg.mixed_precision == PrecisionType.BF16:
                    obj.forward = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)(obj.forward)
                elif self.cfg.mixed_precision == PrecisionType.FP16:
                    obj.forward = torch.autocast(device_type=self.device.type, dtype=torch.float16)(obj.forward)
        return prepared
