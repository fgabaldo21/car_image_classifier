import os

import mat4py
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

from model import model as model_module

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    data = yaml.safe_load(f)


CARS_META_PATH = os.path.join(
    BASE_DIR, "data", "cars196", "car_devkit", "devkit", "cars_meta.mat"
)

name_data = mat4py.loadmat(CARS_META_PATH)
class_names = name_data["class_names"]


MODEL_PATH = os.path.join(BASE_DIR, "app", "ml_model", "model.pth")

device = torch.device("cpu")
model = model_module.Cnn()
torch.save(model.state_dict(), "app/ml_model/model.pth")  # a random model for testing
model.eval()


def predict(img: str) -> str:
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                data["explorations"]["mean"], data["explorations"]["std"]
            ),
        ]
    )

    image = Image.open(img).convert("RGB")

    transformed_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(transformed_image)
        predicted_class = prediction.argmax(dim=1).item()

    return class_names[predicted_class - 1]
