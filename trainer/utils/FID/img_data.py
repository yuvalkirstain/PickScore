import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
from datasets import Dataset as HFDataset


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path, transform=None):
        'Initialization'
        self.file_names = self.get_filenames(path)
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = Image.open(self.file_names[index]).convert('RGB')
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_filenames(self, data_path):
        images = []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                    filename = os.path.join(path, name)
                    if os.path.isfile(filename):
                        images.append(filename)
        return images


class HFImgDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        example = self.dataset[item]
        if self.transform is not None:
            example["image"] = self.transform(example["image"])
        return example["image"]


if __name__ == '__main__':
    path = "/media/twilightsnow/workspace/gan/AttnGAN/output/birds_attn2_2018_06_24_14_52_20/Model/netG_avg_epoch_300"
    batch_size = 16
    dataset = Dataset(path, transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, batch in enumerate(dataloader):
        print(batch)
        break
