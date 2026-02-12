import torch
import torch.nn as nn
import torch.optim as optim

from .dataloader import get_dataloader
from .model import Cnn


def main(num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Cnn().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_f = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloader()

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}")
        model.train()
        running_loss = 0.0

        for _, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = loss_f(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0

            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)

                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy}"
        )

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
