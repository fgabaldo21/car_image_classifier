from torchvision import transforms
import sys
import os
import yaml

with open('./config/config.yaml', 'r') as f:
    data = yaml.safe_load(f)

class Transformer:
    def __init__(self):
        self.augment_transforms = transforms.Compose([
            transforms.Resize((data['image_dimensions']['height'],data['image_dimensions']['width'])),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(data['explorations']['mean'], data['explorations']['std'])
        ])

        self.normal_transforms = transforms.Compose([
            transforms.Resize((data['image_dimensions']['height'],data['image_dimensions']['width'])),
            transforms.ToTensor(),
            transforms.Normalize(data['explorations']['mean'], data['explorations']['std'])
        ])

    def __call__(self, img, augment: bool = False):
        if augment:
            return self.augment_transforms(img)
        else:
            return self.normal_transforms(img)