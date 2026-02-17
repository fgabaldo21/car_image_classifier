import os

import matplotlib.pyplot as plt
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import make_grid

from .transforms import Transformer

with open("./config/config.yaml", "r") as f:
    data = yaml.safe_load(f)


class CarDataset(Dataset):
    def __init__(self, augment: bool = False, indices: list = None):
        self.img_dir = data["paths"]["images"]
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

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]

        img = Image.open(path).convert("RGB")

        img = self.transform(img, self.augment)

        return img, label


def get_dataloader() -> tuple[DataLoader, DataLoader, DataLoader]:
    original_dataset = CarDataset()
    train_idx, val_idx, test_idx = random_split(
        range(len(original_dataset)), [6500, 1000, 644]
    )

    train = CarDataset(augment=True, indices=train_idx.indices)
    val = CarDataset(augment=False, indices=val_idx.indices)
    test = CarDataset(augment=False, indices=test_idx.indices)

    train_loader = DataLoader(
        train,
        batch_size=data["params"]["batch_size"],
        shuffle=True,
        num_workers=data["params"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val,
        batch_size=data["params"]["batch_size"],
        shuffle=False,
        num_workers=data["params"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test,
        batch_size=data["params"]["batch_size"],
        shuffle=False,
        num_workers=data["params"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


# test block
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader()
    imgs, labels = next(iter(train_loader))
    print(f"Batch imgs.shape: {len(imgs)}, labels.shape: {len(labels)}")

    grid = make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.show()
