from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from transforms import Transformer
import os
from PIL import Image
import yaml

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

with open('./config/config.yaml', 'r') as f:
    data = yaml.safe_load(f)

class CarDataset(Dataset):
    def __init__(self, augment=False):
        self.img_dir = data['paths']['images']
        self.transform = Transformer()
        self.augment = augment
        self.samples = []

        for index, label in enumerate(sorted(os.listdir(self.img_dir))):
            class_path = os.path.join(self.img_dir, label)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.lower().endswith((".png", ".jpg", ".bmp")):
                    self.samples.append((os.path.join(class_path, filename), index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        img = Image.open(path).convert("RGB")

        img = self.transform(img, self.augment)

        return img, label

def get_dataloader():
    original_dataset = CarDataset()
    augmented_dataset = CarDataset(augment=True)

    train_split, test = random_split(original_dataset, [4072, 4072])

    train = ConcatDataset([train_split, augmented_dataset])

    train_loader = DataLoader(
        train, 
        batch_size=data['params']['batch_size'],
        shuffle=True,
        num_workers=data['params']['num_workers'])

    test_loader = DataLoader(
        test, 
        batch_size=data['params']['batch_size'],
        shuffle=False,
        num_workers=data['params']['num_workers'])

    return train_loader, test_loader

# test block
if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()
    imgs, labels = next(iter(train_loader))
    print(f"Batch imgs.shape: {len(imgs)}, labels.shape: {len(labels)}")

    grid = make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()