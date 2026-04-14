import csv
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
CAR_DETAILS_PATH = os.path.join(BASE_DIR, "app", "utils", "car_details.csv")

device = torch.device("cpu")
model = model_module.Cnn()
torch.save(model.state_dict(), "app/ml_model/model.pth")  # a random model for testing
model.eval()


def _load_car_details() -> list[dict[str, str]]:
    with open(CAR_DETAILS_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


car_details_data = _load_car_details()


def predict(img: str) -> dict[str, object]:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
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

    class_id = predicted_class + 1
    details = (
        car_details_data[predicted_class]
        if predicted_class < len(car_details_data)
        else None
    )

    return {
        "class_id": class_id,
        "class_name": class_names[predicted_class],
        "details": details,
    }
