from dataclasses import dataclass
import torch
from omegaconf import II
from torch.nn.modules.loss import _Loss


@dataclass
class CLIPCriterionConfig:
    _target_: str = "trainer.criterions.clip_criterion.CLIPCriterion"
    is_distributed: bool = True
    label_0_column_name: str = II("dataset.label_0_column_name")
    label_1_column_name: str = II("dataset.label_1_column_name")

    input_ids_column_name: str = II("dataset.input_ids_column_name")
    pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    pixels_1_column_name: str = II("dataset.pixels_1_column_name")
    num_examples_per_prompt_column_name: str = II("dataset.num_examples_per_prompt_column_name")
    in_batch_negatives: bool = False
    pass


class CLIPCriterion(_Loss):
    def __init__(self, cfg: CLIPCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_features(model, input_ids, pixels_0_values, pixels_1_values):
        all_pixel_values = torch.cat([pixels_0_values, pixels_1_values], dim=0)
        text_features, all_image_features = model(text_inputs=input_ids, image_inputs=all_pixel_values)
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_0_features, image_1_features = all_image_features.chunk(2, dim=0)
        return image_0_features, image_1_features, text_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    def calc_loss(
            self,
            text_features,
            image_0_features,
            image_1_features,
            logit_scale,
            label_0,
            label_1,
            num_examples_per_prompt,
            *args,
            **kwargs
    ):
        device = image_0_features.device

        # gather features
        if self.cfg.is_distributed:
            image_0_features = self.gather_features(image_0_features)
            image_1_features = self.gather_features(image_1_features)
            text_features = self.gather_features(text_features)
            label_0 = self.gather_features(label_0)
            label_1 = self.gather_features(label_1)
            num_examples_per_prompt = self.gather_features(num_examples_per_prompt)

        # calc logits # TODO use local loss as open-clip does
        all_image_features = torch.cat([image_0_features, image_1_features], dim=0)  # (2 * batch_size, dim)
        logits_per_image = logit_scale * all_image_features @ text_features.T
        image_0_logits, image_1_logits = logits_per_image.chunk(2, dim=0)
        text_logits = logit_scale * text_features @ all_image_features.T

        if self.cfg.in_batch_negatives:
            # get labels
            num_images = all_image_features.shape[0]
            image_labels = torch.arange(num_images, device=device, dtype=torch.long)
            image_0_labels, image_1_labels = image_labels.chunk(2, dim=0)
            num_texts = text_features.shape[0]
            text_labels = torch.arange(num_texts, device=device, dtype=torch.long)

            # image loss - we want to increase the logits of the preferred image to the text
            image_0_loss = torch.nn.functional.cross_entropy(image_0_logits, text_labels, reduction="none")
            image_1_loss = torch.nn.functional.cross_entropy(image_1_logits, text_labels, reduction="none")
            # if we have a tie, we will increase both images equally, and average so the image loss of each example is
            # proportional
            image_loss = label_0 * image_0_loss + label_1 * image_1_loss

            # text loss - we want to increase the logits of the text to the preferred image
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, image_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, image_1_labels, reduction="none")

        else:
            text_0_logits, text_1_logits = text_logits.chunk(2, dim=-1)
            index = torch.arange(text_0_logits.shape[0], device=device, dtype=torch.long)
            text_0_logits = text_0_logits[index, index]
            text_1_logits = text_1_logits[index, index]
            text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
            text_0_labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
            text_1_labels = text_0_labels + 1
            text_0_loss = torch.nn.functional.cross_entropy(text_logits, text_0_labels, reduction="none")
            text_1_loss = torch.nn.functional.cross_entropy(text_logits, text_1_labels, reduction="none")

        # if we have a tie we want the logits of for each image to be equal
        text_loss = label_0 * text_0_loss + label_1 * text_1_loss
        # we want the ideal loss to be 0, currently, if there is a tie, it is 0.5 * log(0.5) + 0.5 * log(0.5)
        # so we add log(0.5) to the loss
        is_tie = (label_0 == label_1).float()
        is_tie *= torch.log(torch.tensor(0.5, device=device))
        text_loss += is_tie

        # we average the image and text loss
        if self.cfg.in_batch_negatives:
            loss = (image_loss + text_loss) / 2
        else:
            loss = text_loss

        # some prompts have lots of interactions, we want weight them accordingly
        absolute_example_weight = 1 / num_examples_per_prompt
        denominator = absolute_example_weight.sum()
        weight_per_example = absolute_example_weight / denominator
        loss *= weight_per_example

        loss = loss.sum()
        return loss

    def forward(self, model, batch):
        image_0_features, image_1_features, text_features = self.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name]
        )
        loss = self.calc_loss(
            text_features,
            image_0_features,
            image_1_features,
            model.logit_scale.exp(),
            batch[self.cfg.label_0_column_name],
            batch[self.cfg.label_1_column_name],
            batch[self.cfg.num_examples_per_prompt_column_name],
        )
        return loss
