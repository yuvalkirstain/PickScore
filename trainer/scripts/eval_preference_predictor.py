import numpy as np
from transformers import AutoProcessor, AutoModel
from datasets import load_from_disk, load_dataset
import torch
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from fire import Fire


def open_image(image):
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert("RGB")
    return image


@torch.no_grad()
def infer_example(images, prompt, clip_model, clip_processor, device):
    images = [open_image(image) for image in images]

    image_inputs = clip_processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = clip_processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_embs = clip_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = clip_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = clip_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()


def calc_probs_for_dataset(ds, clip_model, clip_processor, device):
    probs = []
    for example in tqdm(ds):
        prob_0, prob_1 = infer_example(
            [example["jpg_0"], example["jpg_1"]],
            example["caption"],
            clip_model,
            clip_processor,
            device
        )
        probs.append((prob_0, prob_1))
    return probs


def get_label(example):
    if example["label_0"] == 0.5:
        label = "tie"
    elif example["label_0"] == 1:
        label = "0"
    else:
        label = "1"
    return label


def get_pred(prob_0, prob_1, threshold):
    if abs(prob_1 - prob_0) <= threshold:
        pred = "tie"
    elif prob_0 > prob_1:
        pred = "0"
    else:
        pred = "1"
    return pred


def calc_score(label, pred):
    if label == pred:
        score = 1
    elif "tie" in [label, pred]:
        score = 0.5
    else:
        score = 0
    return score


def calc_acc(probs, ds, threshold):
    res = []
    for example, (prob_0, prob_1) in zip(ds, probs):
        label = get_label(example)
        pred = get_pred(prob_0, prob_1, threshold)
        score = calc_score(label, pred)
        res.append(score)
    return sum(res) / len(res)


def calc_scores(ds, thresholds, clip_model, clip_processor, device):
    if isinstance(thresholds, (float, int)):
        thresholds = [thresholds]

    probs = calc_probs_for_dataset(ds, clip_model, clip_processor, device)

    res = []
    for threshold in thresholds:
        acc = calc_acc(probs, ds, threshold)
        res.append(acc)

    return res


def calc_random_scores(ds, thresholds):
    if isinstance(thresholds, (float, int)):
        thresholds = [thresholds]

    total = len(ds)
    num_ties = len(ds.filter(lambda x: x["label_0"] == 0.5))
    num_no_ties = total - num_ties
    random_scores = [((threshold * 1 + ((1 - threshold) / 2)) * num_ties / total) + ((num_no_ties / total) * 0.5) for
                     threshold
                     in thresholds]

    return random_scores


def main(processor_pretrained_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
         model_pretrained_name_or_path: str = "yuvalkirstain/PickScore_v1",
         dataset_name_or_path: str = "yuvalkirstain/pickapic_v1",
         should_load_from_disk: bool = False):
    device = "cuda:0"

    print(f"Loading dataset {dataset_name_or_path}")
    if should_load_from_disk:
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    print(f"Loading model {model_pretrained_name_or_path}")
    clip_processor = AutoProcessor.from_pretrained(processor_pretrained_name_or_path)
    clip_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    validation_split = "validation_unique"
    test_split = "test_unique"

    print("Calculating validation accuracy")
    ours_validation_results = calc_scores(dataset[validation_split], thresholds, clip_model, clip_processor, device)
    best_threshold = np.argmax(ours_validation_results)
    print(ours_validation_results)
    print(f"Best Threshold: {thresholds[best_threshold]} | Acc: {ours_validation_results[best_threshold]}")
    print("Calculating test accuracy")
    test_result = calc_scores(dataset[test_split], thresholds[best_threshold], clip_model, clip_processor, device)
    print(f"Test Acc: {test_result[0]}")


if __name__ == '__main__':
    Fire(main)
