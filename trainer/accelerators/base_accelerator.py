import abc
import hashlib
import json
import math
import os
import shutil
from dataclasses import field, dataclass
from glob import glob
from typing import List, Optional

import datasets
import torch
import transformers
from accelerate.logging import get_logger
from accelerate.utils import set_seed as accelerate_set_seed, PrecisionType
from accelerate.utils.dataclasses import BaseEnum, LoggerType, DynamoBackend
from omegaconf import DictConfig, OmegaConf, II
from tqdm import tqdm

from trainer.accelerators.utils import get_nvidia_smi_gpu_memory_stats_str, print_config, _flatten_dict

logger = get_logger(__name__)

TRAINING_STAGE_PATH = "training_stage.json"


def debug(port):
    logger.info("Connecting to debugger...")
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True)


@dataclass
class DebugConfig:
    activate: bool = False
    port: int = 5900


class TrainingMode(BaseEnum):
    SKIPPING = "skipping"
    TRAINING = "training"


class MetricMode(BaseEnum):
    MAX = "max"
    MIN = "min"


@dataclass
class BaseAcceleratorConfig:
    _target_: str = "trainer.accelerators.base_accelerator.Accelerator"
    output_dir: str = II("output_dir")
    mixed_precision: PrecisionType = PrecisionType.NO
    gradient_accumulation_steps: int = 1
    log_with: Optional[LoggerType] = LoggerType.WANDB
    debug: DebugConfig = DebugConfig()
    seed: int = 42
    resume_from_checkpoint: bool = True
    max_steps: int = 4000
    num_epochs: int = 10
    validate_steps: int = 100
    eval_on_start: bool = True
    project_name: str = "reward"
    max_grad_norm: float = 1.0
    save_steps: int = 100
    metric_name: str = "accuracy"
    metric_mode: MetricMode = MetricMode.MAX
    limit_num_checkpoints: int = 1
    save_only_if_best: bool = True
    dynamo_backend: DynamoBackend = DynamoBackend.NO
    keep_best_ckpts: bool = True


class BaseAccelerator(abc.ABC):

    def __init__(self, cfg: BaseAcceleratorConfig):
        self.cfg = cfg
        self.accelerator = None
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.step_loss = 0.0
        self.lr = None
        self.metrics = {}
        self.progress_bar = None
        self.mode = TrainingMode.TRAINING
        self.num_update_steps_per_epoch = None
        self.num_steps_per_epoch = None

    def post_init(self):
        self.set_seed()
        self.debug()
        logger.info(f"Initialized accelerator: rank={self.accelerator.process_index}", main_process_only=False)
        self.set_logging_level()

    def set_logging_level(self):
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    def debug(self):
        if self.accelerator.is_main_process and self.cfg.debug.activate:
            debug(self.cfg.debug.port)

    def set_seed(self):
        logger.info(f"Setting seed {self.cfg.seed}")
        accelerate_set_seed(self.cfg.seed, device_specific=True)

    def prepare(self, *args, device_placement=None):
        return self.accelerator.prepare(*args, device_placement=device_placement)

    def get_latest_checkpoint(self):
        all_ckpts = list(glob(os.path.join(self.cfg.output_dir, "checkpoint-*")))
        if len(all_ckpts) == 0:
            return
        all_ckpts.sort(key=os.path.getctime)
        if "final" in all_ckpts[-1]:
            all_ckpts.pop()
        return all_ckpts[-1] if len(all_ckpts) > 0 else None

    def load_state_if_needed(self):
        if not self.cfg.resume_from_checkpoint:
            return
        ckpt_path = self.get_latest_checkpoint()

        if ckpt_path is None:
            logger.info("No checkpoint found, training from scratch")
            return

        stage = json.load(open(os.path.join(ckpt_path, TRAINING_STAGE_PATH)))
        self.epoch, self.step, self.global_step, self.metrics = stage["epoch"], stage["step"], stage["global_step"], \
            stage["metrics"]
        logger.info(
            f"Resuming from checkpoint: {ckpt_path} | epoch={self.epoch} step={self.step} gstep={self.global_step}")
        self.accelerator.load_state(ckpt_path)
        logger.info("Checkpoint loaded")

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def num_processes(self):
        return self.accelerator.num_processes

    def pre_training_log(self, cfg: DictConfig):
        total_batch_size = cfg.dataset.batch_size * self.num_processes * self.cfg.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {cfg.dataset.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}")
        logger.info(f"  Total warmup steps = {cfg.lr_scheduler.lr_warmup_steps}")
        logger.info(f"  Total training steps = {self.cfg.max_steps * self.cfg.gradient_accumulation_steps}")
        logger.info(f"  Total epochs = {self.cfg.num_epochs}")
        logger.info(f"  Steps per epoch = {self.num_steps_per_epoch}")
        logger.info(f"  Update steps per epoch = {self.num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {self.cfg.max_steps}")
        logger.info(f"  Mixed precision = {self.cfg.mixed_precision}")
        logger.info(f"  World size = {self.accelerator.num_processes}")

    def init_training(self, cfg: DictConfig):
        if self.is_main_process:
            yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
            log_cfg = _flatten_dict(OmegaConf.create(yaml))
            logger.info("Initializing trackers")
            self.accelerator.init_trackers(self.cfg.project_name, log_cfg)
            logger.info("Training config:")
            print_config(cfg)
        logger.info(get_nvidia_smi_gpu_memory_stats_str())
        self.pre_training_log(cfg)
        self.progress_bar = tqdm(range(self.cfg.max_steps * self.cfg.gradient_accumulation_steps), disable=not self.accelerator.is_main_process)
        self.progress_bar.set_description("Steps")

    def should_skip(self, epoch, step):
        should = epoch < self.epoch or (epoch == self.epoch and step < self.step)
        if should:
            self.mode = TrainingMode.SKIPPING
            self.progress_bar.set_postfix(**{"status": TrainingMode.SKIPPING})
        else:
            self.mode = TrainingMode.TRAINING
        return should

    def update_progbar_step(self):
        self.progress_bar.update(1)

    def log(self, data):
        if self.is_main_process:
            self.accelerator.log(data, step=self.global_step)

    def recalc_train_length_after_prepare(self, num_batches):
        num_update_steps_per_epoch = math.ceil(num_batches / self.cfg.gradient_accumulation_steps)
        if self.cfg.max_steps is None:
            self.cfg.max_steps = self.cfg.num_epochs * num_update_steps_per_epoch
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.num_steps_per_epoch = num_batches
        self.cfg.num_epochs = math.ceil(self.cfg.max_steps / num_update_steps_per_epoch)

    def accumulate(self, model):
        return self.accelerator.accumulate(model)

    def gather(self, data):
        return self.accelerator.gather(data)

    @property
    def sync_gradients(self):
        return self.accelerator.sync_gradients

    def update_step_loss(self, loss):
        self.step_loss = loss

    def update_global_step(self, loss):
        self.global_step += 1
        self.log({
            "lr": self.lr,
            "step": self.step,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss": loss,
        })

    def get_allocated_cuda_memory(self):
        return round(torch.cuda.max_memory_allocated(self.accelerator.device) / 1024 / 1024 / 1024, 2)

    def update_step(self, loss, lr):
        self.step += 1
        self.lr = lr
        logs = {
            "stl": loss,
            "gstl": loss,
            "mem": self.get_allocated_cuda_memory(),
            "st": self.step,
            "ep": self.epoch,
            "gst": self.global_step,
            "lr": self.lr,
        }
        self.progress_bar.set_postfix(**logs)
        self.update_progbar_step()

    def wait_for_everyone(self):
        self.accelerator.wait_for_everyone()

    def update_epoch(self):
        if self.mode == TrainingMode.SKIPPING:
            return
        logger.info(f"Epoch {self.epoch} finished")
        self.epoch += 1
        self.step = 0

    def update_metrics(self, metrics):
        self.metrics.update(metrics)
        logger.info(f"Metrics: {self.metrics}")
        self.log(metrics)

    def end_training(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def unwrap_and_save(self, model):
        if not self.is_main_process:
            return
        model = self.accelerator.unwrap_model(model)
        save_dir = os.path.join(self.cfg.output_dir, f"checkpoint-final")
        logger.info(f"Saving final checkpoint to {save_dir}")
        model.save(save_dir)
        self.save_training_stage(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")

    def should_end(self):
        return self.global_step >= self.cfg.max_steps

    def backward(self, loss):
        self.accelerator.backward(loss)

    def clip_grad_norm_(self, params):
        self.accelerator.clip_grad_norm_(params, self.cfg.max_grad_norm)

    def should_eval(self):
        if not self.mode == TrainingMode.TRAINING:
            return False
        if self.step == 0 and self.global_step == 0 and self.cfg.eval_on_start:
            return True
        if self.global_step > 0 and self.sync_gradients and self.global_step % self.cfg.validate_steps == 0:
            return True
        return False

    def should_save(self):
        return self.sync_gradients and self.global_step > 0 and self.cfg.save_steps > 0 and self.global_step % self.cfg.save_steps == 0

    @property
    def training_stage(self):
        return {
            "epoch": self.epoch,
            "step": self.step,
            "global_step": self.global_step,
            "step_loss": self.step_loss,
            "lr": self.lr,
            "metrics": self.metrics,
        }

    def save_training_stage(self, save_dir):
        json.dump(self.training_stage, open(os.path.join(save_dir, TRAINING_STAGE_PATH), "w"), indent=4)

    def save_checkpoint(self):
        if self.cfg.save_only_if_best:
            all_ckpts = self.get_all_ckpts()
            for ckpt in all_ckpts:
                training_stage = json.load(open(os.path.join(ckpt, TRAINING_STAGE_PATH)))
                metric_val = training_stage["metrics"][self.cfg.metric_name]
                cur_metric_val = self.training_stage["metrics"][self.cfg.metric_name]
                if (self.cfg.metric_mode == MetricMode.MIN and metric_val < cur_metric_val) or \
                        (self.cfg.metric_mode == MetricMode.MAX and metric_val > cur_metric_val):
                    logger.info(
                        f"Metric {self.cfg.metric_name}={cur_metric_val} is not better than {metric_val} of {ckpt}, skipping checkpoint")
                    return
        self.cleanup_checkpoints()
        self.accelerator.wait_for_everyone()
        save_dir = os.path.join(self.cfg.output_dir, f"checkpoint-gstep{self.global_step}")
        logger.info(f"Saving checkpoint to {save_dir}")
        self.accelerator.save_state(save_dir)
        self.save_training_stage(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")

    @property
    def gradient_state(self):
        return self.accelerator.gradient_state

    def get_all_ckpts(self):
        return list(glob(os.path.join(self.cfg.output_dir, f"checkpoint-*")))

    def load_best_checkpoint(self):
        all_ckpts = self.get_all_ckpts()
        if not self.cfg.keep_best_ckpts:
            all_ckpts.sort(key=os.path.getctime, reverse=True)
            logger.info(f"Returning the most recent checkpoint: {all_ckpts[0]}")
            return all_ckpts[0]
        logger.info(f"Found {len(all_ckpts)} checkpoints in {self.cfg.output_dir}")
        logger.info(all_ckpts)
        if len(all_ckpts) == 0:
            logger.info(f"No checkpoint found in {self.cfg.output_dir} to load. Keeping current model.")
            return
        best_ckpt, best_metric_val = None, math.inf if self.cfg.metric_mode == MetricMode.MIN else -math.inf
        for ckpt in all_ckpts:
            training_stage = json.load(open(os.path.join(ckpt, TRAINING_STAGE_PATH)))
            metric_val = training_stage["metrics"][self.cfg.metric_name]
            if (self.cfg.metric_mode == MetricMode.MIN and metric_val < best_metric_val) or \
                    (self.cfg.metric_mode == MetricMode.MAX and metric_val > best_metric_val):
                best_ckpt, best_metric_val = ckpt, metric_val
        logger.info(f"Loading best checkpoint from {best_ckpt} with metric {self.cfg.metric_name}={best_metric_val}")
        self.accelerator.load_state(best_ckpt)

    @property
    def device(self):
        return self.accelerator.device

    def cleanup_checkpoints(self):
        if self.cfg.limit_num_checkpoints <= 0 or not self.accelerator.is_main_process:
            logger.info(f"Not cleaning up checkpoints as limit_num_checkpoints={self.cfg.limit_num_checkpoints}")
            return

        all_ckpts = self.get_all_ckpts()
        if len(all_ckpts) <= self.cfg.limit_num_checkpoints:
            logger.info(f"Not cleaning up checkpoints as only {len(all_ckpts)} checkpoints found")
            return

        logger.info(f"Found {len(all_ckpts)} checkpoints in {self.cfg.output_dir}")
        ckpts_to_delete = self.get_ckpts_to_delete()
        ckpts_to_delete.sort(key=os.path.getctime)

        ckpts_to_delete = ckpts_to_delete[:-1]
        for ckpt in ckpts_to_delete:
            logger.info(f"Deleting checkpoint {ckpt}")
            shutil.rmtree(ckpt)

    def get_ckpts_to_delete(self):
        all_ckpts = self.get_all_ckpts()
        if self.cfg.keep_best_ckpts:
            metric_vals = []
            for ckpt in all_ckpts:
                training_stage = json.load(open(os.path.join(ckpt, TRAINING_STAGE_PATH)))
                metric_val = training_stage["metrics"][self.cfg.metric_name]
                metric_vals.append(metric_val)
            metric_ckpt = list(zip(metric_vals, all_ckpts))
            metric_ckpt.sort(key=lambda x: x[0], reverse=self.cfg.metric_mode == MetricMode.MAX)
            ckpts_to_delete = [ckpt for _, ckpt in metric_ckpt[self.cfg.limit_num_checkpoints:]]
        else:
            all_ckpts.sort(key=os.path.getctime, reverse=True)
            ckpts_to_delete = all_ckpts[self.cfg.limit_num_checkpoints:]
        return ckpts_to_delete