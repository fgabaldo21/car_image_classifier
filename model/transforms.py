from torchvision import dataset, transforms

import sys
import os
import yaml

with open('./config/config.yaml', 'r') as f:
    data = yaml.safe_load(f)

class Transformer:
    def __init__(self):
        self.classes_to_augment = explore.result_list

        self.augment_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(explore.avg_mean, explore.avg_std)
        ])

        self.normal_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(explore.avg_mean, explore.avg_std) # tensor([[0.4691, 0.4588, 0.4540]], tensor([[0.2596, 0.2582, 0.2631]]
        ])