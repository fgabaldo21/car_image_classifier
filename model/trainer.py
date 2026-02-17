import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .dataloader import get_dataloader
from .model import Cnn

with open("./config/config.yaml", "r") as f:
    data = yaml.safe_load(f)


def main(
    num_epochs: int = 10,
    learning_rate: float = 0.01,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Cnn().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs
    )
    loss_f = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = get_dataloader()

    best_val_accuracy = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for image, label in tqdm(train_loader, desc="Training"):
            image, label = (
                image.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )

            output = model(image)
            loss = loss_f(output, label)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            for image, label in tqdm(val_loader, desc="Validation"):
                image, label = (
                    image.to(device, non_blocking=True),
                    label.to(device, non_blocking=True),
                )

                output = model(image)
                loss = loss_f(output, label)

                val_loss += loss.item()

                _, predicted = torch.max(output, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        scheduler.step()

        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss: {avg_val_loss:.4f}, Val accuracy: {val_accuracy:.4f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model! Vall accuracy: {val_accuracy:.4f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvements in {patience} epochs. Early stopping")
            break

    print("\n" + "=" * 50)
    print("Training complete! Evaluating on test set...")
    print("=" * 50)

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for image, label in tqdm(test_loader, desc="Testing"):
            image, label = (
                image.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {test_accuracy:.4f}%")


if __name__ == "__main__":
    main(
        num_epochs=data["params"]["num_epochs"],
        learning_rate=data["params"]["learning_rate"],
    )
