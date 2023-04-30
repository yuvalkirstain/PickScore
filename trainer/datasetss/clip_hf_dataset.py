from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig

logger = get_logger(__name__)


def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


@dataclass
class ProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")


@dataclass
class CLIPHFDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.clip_hf_dataset.CLIPHFDataset"
    dataset_name: str = "yuvalkirstain/pickapic_v1"
    dataset_config_name: str = "null"

    from_disk: bool = False
    train_split_name: str = "train"
    valid_split_name: str = "validation_unique"
    test_split_name: str = "test_unique"
    cache_dir: Optional[str] = None

    caption_column_name: str = "caption"
    input_ids_column_name: str = "input_ids"
    image_0_column_name: str = "jpg_0"
    image_1_column_name: str = "jpg_1"
    label_0_column_name: str = "label_0"
    label_1_column_name: str = "label_1"
    are_different_column_name: str = "are_different"
    has_label_column_name: str = "has_label"

    pixels_0_column_name: str = "pixel_values_0"
    pixels_1_column_name: str = "pixel_values_1"

    num_examples_per_prompt_column_name: str = "num_example_per_prompt"

    keep_only_different: bool = False
    keep_only_with_label: bool = False
    keep_only_with_label_in_non_train: bool = True

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

    only_on_best: bool = False


class CLIPHFDataset(BaseDataset):

    def __init__(self, cfg: CLIPHFDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        logger.info(f"Loading {self.split} dataset")

        self.dataset = self.load_hf_dataset(self.split)
        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")

        if self.cfg.keep_only_different:
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.are_different_column_name])

        if self.cfg.keep_only_with_label:
            logger.info(f"Keeping only examples with label")
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.has_label_column_name])
            logger.info(f"Kept {len(self.dataset)} examples from {self.split} dataset")
        elif self.cfg.keep_only_with_label_in_non_train and self.split != self.cfg.train_split_name:
            logger.info(f"Keeping only examples with label in {self.split} split")
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.has_label_column_name])
            logger.info(f"Kept {len(self.dataset)} examples from {self.split} dataset")

        if self.cfg.limit_examples_per_prompt > 0:
            logger.info(f"Limiting examples per prompt to {self.cfg.limit_examples_per_prompt}")
            df = self.dataset.to_pandas()
            df = df.drop('__index_level_0__', axis=1)
            logger.info(f"Loaded {len(df)} examples from {self.split} dataset")
            df = df.groupby(self.cfg.caption_column_name).head(self.cfg.limit_examples_per_prompt)
            logger.info(f"Kept {len(df)} examples from {self.split} dataset")
            self.dataset = Dataset.from_pandas(df)

        if self.cfg.only_on_best and self.split == self.cfg.train_split_name:
            logger.info(f"Keeping only best examples for training")
            train_dataset = self.dataset.remove_columns([self.cfg.image_0_column_name, self.cfg.image_1_column_name])
            df = train_dataset.to_pandas()
            df = df[df[self.cfg.has_label_column_name] == 1]
            image_0_wins_df = df[df[self.cfg.label_0_column_name] == 1]
            image_1_wins_df = df[df[self.cfg.label_0_column_name] == 0]
            bad_image_0_to_good_image_1 = dict(zip(image_1_wins_df.image_0_uid, image_1_wins_df.image_1_uid))
            bad_image_1_to_good_image_0 = dict(zip(image_0_wins_df.image_1_uid, image_0_wins_df.image_0_uid))
            bad_images_uids2good_images_uids = bad_image_0_to_good_image_1 | bad_image_1_to_good_image_0
            image_0_uid2image_col_name = dict(zip(df.image_0_uid, [self.cfg.image_0_column_name] * len(df.image_0_uid)))
            image_1_uid2image_col_name = dict(zip(df.image_1_uid, [self.cfg.image_1_column_name] * len(df.image_1_uid)))
            uid2image_col_name = image_0_uid2image_col_name | image_1_uid2image_col_name

            bad_uids = set()
            for bad_image, good_image in bad_images_uids2good_images_uids.items():
                cur_good = {bad_image}
                while good_image in bad_images_uids2good_images_uids:
                    if good_image in cur_good:
                        bad_uids.add(bad_image)
                        break
                    cur_good.add(good_image)
                    good_image = bad_images_uids2good_images_uids[good_image]
                bad_images_uids2good_images_uids[bad_image] = good_image

            df = df[~(df.image_0_uid.isin(bad_uids) | df.image_1_uid.isin(bad_uids))]
            keep_ids = df.index.tolist()
            self.dataset = self.dataset.select(keep_ids)
            new_ids = list(range(len(df)))
            uid2index = dict(zip(df.image_0_uid, new_ids)) | dict(zip(df.image_1_uid, new_ids))
            logger.info(f"Kept only {len(self.dataset)} best examples for training")
            self.bad_images_uids2good_images_uids = bad_images_uids2good_images_uids
            self.uid2index = uid2index
            self.uid2image_col_name = uid2image_col_name

        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")

        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

    def load_hf_dataset(self, split: str) -> Dataset:
        if self.cfg.from_disk:
            dataset = load_from_disk(self.cfg.dataset_name)[split]
        else:
            dataset = load_dataset(
                self.cfg.dataset_name,
                self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                split=split
            )
        return dataset

    def tokenize(self, example):
        caption = example[self.cfg.caption_column_name]
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids

    def process_image(self, image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        image = image.convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values

    def __getitem__(self, idx):
        example = self.dataset[idx]

        if self.cfg.only_on_best and self.split == self.cfg.train_split_name:
            if example[self.cfg.label_0_column_name]:
                bad_image_uid = example["image_1_uid"]
                good_image_column_name = self.cfg.image_0_column_name
            else:
                bad_image_uid = example["image_0_uid"]
                good_image_column_name = self.cfg.image_1_column_name
            good_image_uid = self.bad_images_uids2good_images_uids[bad_image_uid]
            good_image_index = self.uid2index[good_image_uid]
            example[good_image_column_name] = self.dataset[good_image_index][self.uid2image_col_name[good_image_uid]]

        input_ids = self.tokenize(example)

        pixel_0_values = self.process_image(example[self.cfg.image_0_column_name])
        pixel_1_values = self.process_image(example[self.cfg.image_1_column_name])

        item = {
            self.cfg.input_ids_column_name: input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.label_0_column_name: torch.tensor(example[self.cfg.label_0_column_name])[None],
            self.cfg.label_1_column_name: torch.tensor(example[self.cfg.label_1_column_name])[None],
            self.cfg.num_examples_per_prompt_column_name: torch.tensor(example[self.cfg.num_examples_per_prompt_column_name])[None],
        }
        return item

    def collate_fn(self, batch):
        input_ids = simple_collate(batch, self.cfg.input_ids_column_name)
        pixel_0_values = simple_collate(batch, self.cfg.pixels_0_column_name)
        pixel_1_values = simple_collate(batch, self.cfg.pixels_1_column_name)
        label_0 = simple_collate(batch, self.cfg.label_0_column_name)
        label_1 = simple_collate(batch, self.cfg.label_1_column_name)
        num_examples_per_prompt = simple_collate(batch, self.cfg.num_examples_per_prompt_column_name)

        pixel_0_values = pixel_0_values.to(memory_format=torch.contiguous_format).float()
        pixel_1_values = pixel_1_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            self.cfg.input_ids_column_name: input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.label_0_column_name: label_0,
            self.cfg.label_1_column_name: label_1,
            self.cfg.num_examples_per_prompt_column_name: num_examples_per_prompt,
        }
        return collated

    def __len__(self):
        return len(self.dataset)
