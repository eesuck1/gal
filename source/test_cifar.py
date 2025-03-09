import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from source.gal import GAL
from source.models import ResNet, BasicBlock
from source.utils import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

def load_data() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader

def train(local_model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.AdamW, model_name: str):
    local_model.train()

    local_losses = []
    local_accuracy = []

    for epoch, (images, labels) in enumerate(train_loader):
        print(f"[{epoch}]")

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = local_model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        local_losses.append(loss.item())
        _, predicted = outputs.max(1)

        correct = predicted.eq(labels).sum().item()
        local_accuracy.append(correct / BATCH_SIZE)

        if epoch % 25 == 0:
            with torch.no_grad():
                l = models["GAL"].activation

                p = torch.cat([l._p_l, l._p_r])
                x = torch.linspace(-10.0, 10.0, 1000).to(device)

                plt.plot(x.cpu(), l(x).cpu(), linewidth=1.0)

                for p_i in p:
                    plt.axvline(p_i.cpu(), linewidth=1.0, linestyle="--", color="tab:green")

                plt.grid()
                plt.show()

        if epoch > 200:
            break

    torch.save(local_model.state_dict(), f"./models/{model_name}.pt")

    return local_losses, local_accuracy

if __name__ == '__main__':
    torch.manual_seed(0)

    blocks = [2, 2, 2, 2]

    models = {
        "GAL": ResNet(BasicBlock, blocks, activation=lambda: GAL(1, device), use_norm=True),
        "ReLU": ResNet(BasicBlock, blocks, activation=nn.ReLU, use_norm=True),
        "Tanh": ResNet(BasicBlock, blocks, activation=nn.Tanh, use_norm=True),
    }

    for model in models.values():
        model.to(device)

    for name, model in models.items():
        print(f"{name}: {count_parameters(model)}")

    optimizers = {name: optim.AdamW(model.parameters(), lr=3e-4) for name, model in models.items()}
    train_losses = {name: [] for name in models}
    train_accuracy = {name: [] for name in models}
    val_losses = {name: [] for name in models}

    loss_function = nn.CrossEntropyLoss()
    train_data, test_data = load_data()

    for name, model in models.items():
        losses, accuracy = train(model, train_data, loss_function, optimizers[name], name)

        train_losses[name] = losses
        train_accuracy[name] = accuracy

    figure, axes = plt.subplots(2, 1, figsize=(7, 7))

    for name in models:
        axes[0].plot(train_losses[name], label=name, linewidth=1.0)
        axes[1].plot(train_accuracy[name], label=name, linewidth=1.0)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid()

    plt.show()
