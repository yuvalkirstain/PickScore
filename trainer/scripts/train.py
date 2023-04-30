import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_dataloaders(cfg: DictConfig) -> Any:
    dataloaders = {}
    for split in [cfg.train_split_name, cfg.valid_split_name, cfg.test_split_name]:
        dataset = instantiate_with_cfg(cfg, split=split)
        should_shuffle = split == cfg.train_split_name
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            shuffle=should_shuffle,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.num_workers
        )
    return dataloaders


def load_optimizer(cfg: DictConfig, model: nn.Module):
    optimizer = instantiate(cfg, model=model)
    return optimizer


def load_scheduler(cfg: DictConfig, optimizer):
    scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
    return scheduler


def load_task(cfg: DictConfig, accelerator: BaseAccelerator):
    task = instantiate_with_cfg(cfg, accelerator=accelerator)
    return task


def verify_or_write_config(cfg: TrainerConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    yaml_path = os.path.join(cfg.output_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        OmegaConf.save(cfg, yaml_path, resolve=True)
    with open(yaml_path) as f:
        existing_config = f.read()
    if existing_config != OmegaConf.to_yaml(cfg, resolve=True):
        raise ValueError(f"Config was not saved correctly - {yaml_path}")
    logger.info(f"Config can be found in {yaml_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: TrainerConfig) -> None:
    accelerator = instantiate_with_cfg(cfg.accelerator)

    if cfg.debug.activate and accelerator.is_main_process:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)

    if accelerator.is_main_process:
        verify_or_write_config(cfg)

    logger.info(f"Loading task")
    task = load_task(cfg.task, accelerator)
    logger.info(f"Loading model")
    model = instantiate_with_cfg(cfg.model)
    logger.info(f"Loading criterion")
    criterion = instantiate_with_cfg(cfg.criterion)
    logger.info(f"Loading optimizer")
    optimizer = load_optimizer(cfg.optimizer, model)
    logger.info(f"Loading lr scheduler")
    lr_scheduler = load_scheduler(cfg.lr_scheduler, optimizer)
    logger.info(f"Loading dataloaders")
    split2dataloader = load_dataloaders(cfg.dataset)

    dataloaders = list(split2dataloader.values())
    model, optimizer, lr_scheduler, *dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, *dataloaders)
    split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))

    accelerator.load_state_if_needed()

    accelerator.recalc_train_length_after_prepare(len(split2dataloader[cfg.dataset.train_split_name]))

    accelerator.init_training(cfg)

    def evaluate():
        model.eval()
        end_of_train_dataloader = accelerator.gradient_state.end_of_dataloader
        logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
        metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
        accelerator.update_metrics(metrics)
        accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

    logger.info(f"task: {task.__class__.__name__}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"num. model params: {int(sum(p.numel() for p in model.parameters()) // 1e6)}M")
    logger.info(
        f"num. model trainable params: {int(sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6)}M")
    logger.info(f"criterion: {criterion.__class__.__name__}")
    logger.info(f"num. train examples: {len(split2dataloader[cfg.dataset.train_split_name].dataset)}")
    logger.info(f"num. valid examples: {len(split2dataloader[cfg.dataset.valid_split_name].dataset)}")
    logger.info(f"num. test examples: {len(split2dataloader[cfg.dataset.test_split_name].dataset)}")

    for epoch in range(accelerator.cfg.num_epochs):
        train_loss, lr = 0.0, 0.0
        for step, batch in enumerate(split2dataloader[cfg.dataset.train_split_name]):
            if accelerator.should_skip(epoch, step):
                accelerator.update_progbar_step()
                continue

            if accelerator.should_eval():
                evaluate()

            if accelerator.should_save():
                accelerator.save_checkpoint()

            model.train()

            with accelerator.accumulate(model):
                loss = task.train_step(model, criterion, batch)
                avg_loss = accelerator.gather(loss).mean().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters())

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss += avg_loss / accelerator.cfg.gradient_accumulation_steps

            if accelerator.sync_gradients:
                accelerator.update_global_step(train_loss)
                train_loss = 0.0

            if accelerator.global_step > 0:
                lr = lr_scheduler.get_last_lr()[0]

            accelerator.update_step(avg_loss, lr)

            if accelerator.should_end():
                evaluate()
                accelerator.save_checkpoint()
                break

        if accelerator.should_end():
            break

        accelerator.update_epoch()

    accelerator.wait_for_everyone()
    accelerator.load_best_checkpoint()
    logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
    metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
    accelerator.update_metrics(metrics)
    logger.info(f"*** Evaluating {cfg.dataset.test_split_name} ***")
    metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.test_split_name])
    metrics = {f"{cfg.dataset.test_split_name}_{k}": v for k, v in metrics.items()}
    accelerator.update_metrics(metrics)
    accelerator.unwrap_and_save(model)
    accelerator.end_training()


if __name__ == '__main__':
    main()
