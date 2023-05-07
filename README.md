# PickScore
This repository contains the code for the paper [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569). 

We also open-source the [Pick-a-Pic dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) and [PickScore model](https://huggingface.co/yuvalkirstain/PickScore_v1). We encourage readers to experiment with the [Pick-a-Pic's web application](https://pickapic.io/) and contribute to the dataset.

## Installation
Create a virual env and download torch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

and then install the rest of the requirements:
```bash
pip install -r requirements.txt
pip install -e .
```

Or download each package separately depending on your needs
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.27.3 

# Only required for training
pip install git+https://github.com/huggingface/accelerate.git@d1aa558119859c4b205a324afabaecabd9ef375e
pip install deepspeed==0.8.3
pip install datasets==2.10.1
pip install hydra-core==1.3.2 
pip install rich==13.3.2
pip install wandb==0.12.21
pip install -e .

# Only required for training on slurm
pip install submitit==1.4.5

# Only required for evaluation
pip install fire==0.4.0
```



## Inference with PickScore
We display here an example for running inference with PickScore as a preference predictor:
```python
# import
from transformers import AutoProcessor, AutoModel
from PIL import Image

# load model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

pil_images = [Image.open("my_amazing_images/1.jpg"), Image.open("my_amazing_images/2.jpg")]
prompt = "fantastic, increadible prompt"
print(calc_probs(prompt, pil_images))
```

## Download the Pick-a-Pic Dataset

It took me about 30 minutes to download the dataset which takes about 190GB of disk space. Simply run: 
```bash
from datasets import load_dataset
dataset = load_dataset("yuvalkirstain/pickapic_v1", num_proc=64)
```

Please note that the dataset has more half-a-million images, so you can start by downloading the validation split (add `streaming=True` to avoid downloading the entire dataset) or the version without images (only urls of images):
```python
dataset = load_dataset("yuvalkirstain/pickapic_v1_no_images")
```

## Train PickScore from Scratch
You might want to download the dataset before training to save compute budget. 
Training here is done on 8 A100 GPUs and takes about 40 minutes.

### Locally
```bash
accelerate launch --dynamo_backend no --gpu_ids all --num_processes 8  --num_machines 1 --use_deepspeed trainer/scripts/train.py +experiment=clip_h output_dir=output```
```

### Slurm
```bash
python trainer/slurm_scripts/slurm_train.py +slurm=stability 'slurm.cmd="+experiment=clip_h"'
```

## Test PickScore on Pick-a-Pic
```bash
 python trainer/scripts/eval_preference_predictor.py
```

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{Kirstain2023PickaPicAO,
  title={Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation},
  author={Yuval Kirstain and Adam Polyak and Uriel Singer and Shahbuland Matiana and Joe Penna and Omer Levy},
  year={2023}
}
```
