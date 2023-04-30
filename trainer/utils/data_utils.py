import logging
from glob import glob
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk

logger = logging.getLogger(__name__)


def parquet2dataset(parquet_path: str):
    datasets = []
    for path in sorted(glob(f"{parquet_path}/*.parquet")):
        datasets.append(load_dataset("parquet", data_files=path)["train"])
    dataset = concatenate_datasets(datasets)
    return dataset


def bytes2image(bytes: bytes):
    image = Image.open(BytesIO(bytes))
    image = image.convert("RGB")
    return image


def dataset2images(dataset, pool, col):
    image_bytes = dataset[col]
    images = list(tqdm(pool.imap(bytes2image, image_bytes), total=len(image_bytes)))
    return images
