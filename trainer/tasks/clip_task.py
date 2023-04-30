import collections
from dataclasses import dataclass

import torch
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from omegaconf import II
from transformers import AutoTokenizer

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask

logger = get_logger(__name__)


@dataclass
class CLIPTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.clip_task.CLIPTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    label_0_column_name: str = II("dataset.label_0_column_name")
    label_1_column_name: str = II("dataset.label_1_column_name")

    input_ids_column_name: str = II("dataset.input_ids_column_name")
    pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    pixels_1_column_name: str = II("dataset.pixels_1_column_name")


def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class CLIPTask(BaseTask):
    def __init__(self, cfg: CLIPTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.cfg = cfg

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    @staticmethod
    def features2probs(model, text_features, image_0_features, image_1_features):
        image_0_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        image_0_probs, image_1_probs = probs[:, 0], probs[:, 1]
        return image_0_probs, image_1_probs

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        image_0_features, image_1_features, text_features = criterion.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name]
        )
        return self.features2probs(model, text_features, image_0_features, image_1_features)

    @staticmethod
    def pixel_values_to_pil_images(pixel_values):
        images = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        return images

    def run_inference(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Running clip score...")
        for batch in dataloader:
            image_0_probs, image_1_probs = self.valid_step(model, criterion, batch)
            agree_on_0 = (image_0_probs > image_1_probs) * batch[self.cfg.label_0_column_name]
            agree_on_1 = (image_0_probs < image_1_probs) * batch[self.cfg.label_1_column_name]
            is_correct = agree_on_0 + agree_on_1
            eval_dict["is_correct"] += is_correct.tolist()
            eval_dict["captions"] += self.tokenizer.batch_decode(
                batch[self.cfg.input_ids_column_name],
                skip_special_tokens=True
            )
            eval_dict["image_0"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_0_column_name])
            eval_dict["image_1"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_1_column_name])
            eval_dict["prob_0"] += image_0_probs.tolist()
            eval_dict["prob_1"] += image_1_probs.tolist()

            eval_dict["label_0"] += batch[self.cfg.label_0_column_name].tolist()
            eval_dict["label_1"] += batch[self.cfg.label_1_column_name].tolist()

        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_inference(model, criterion, dataloader)
        eval_dict = self.gather_dict(eval_dict)
        metrics = {
            "accuracy": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"]),
            "num_samples": len(eval_dict["is_correct"])
        }
        if LoggerType.WANDB == self.accelerator.cfg.log_with:
            self.log_to_wandb(eval_dict)
        return metrics
