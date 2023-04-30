from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel

from torch import nn

from trainer.models.base_model import BaseModelConfig


@dataclass
class ClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model.CLIPModel"
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"


class CLIPModel(nn.Module):
    def __init__(self, cfg: ClipModelConfig):
        super().__init__()
        self.model = HFCLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def forward(self, text_inputs=None, image_inputs=None):
        outputs = ()
        if text_inputs is not None:
            outputs += self.model.get_text_features(text_inputs),
        if image_inputs is not None:
            outputs += self.model.get_image_features(image_inputs),
        return outputs


    @property
    def logit_scale(self):
        return self.model.logit_scale

    def save(self, path):
        self.model.save_pretrained(path)

