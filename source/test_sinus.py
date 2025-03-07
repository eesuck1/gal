import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from source.gal import GAL
from source.models import SinModel
from source.utils import count_parameters

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

EPOCHS = 100
BATCH_SIZE = 256
BATCHES = 50
DATA_STEPS = BATCH_SIZE * BATCHES

if __name__ == '__main__':
    torch.manual_seed(0)

    models = {
        "GAL": SinModel(lambda: GAL(3), device, 5),
        "ReLU": SinModel(nn.ReLU, device, 7),
        "Tanh": SinModel(nn.Tanh, device, 7)
    }

    for name, model in models.items():
        print(f"{name}: {count_parameters(model)}")

    optimizers = {name: optim.AdamW(model.parameters(), lr=1e-3) for name, model in models.items()}
    train_losses = {name: [] for name in models}
    val_losses = {name: [] for name in models}
    loss_function = nn.MSELoss()

    x = torch.linspace(-2.0 * torch.pi, 2.0 * torch.pi, DATA_STEPS, device=device)
    y = torch.sin(x).to(device)

    for epoch in range(EPOCHS):
        indexes = torch.randperm(DATA_STEPS)

        data = x[indexes].view(BATCHES, BATCH_SIZE, 1)
        labels = y[indexes].view(BATCHES, BATCH_SIZE, 1)

        for sample, label in zip(data, labels):
            for name, model in models.items():
                out = model(sample)

                loss = loss_function(out, label)

                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()

                train_losses[name].append(loss.item())

        print(f"[{epoch + 1}/{EPOCHS}]")

    for name in models:
        plt.plot(train_losses[name], label=name, linewidth=1.0)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    with torch.no_grad():
        for name, model in models.items():
            plt.plot(x.cpu(), model(x.view(-1, 1)).cpu(), label=name, linewidth=1.0)

        plt.plot(x.cpu(), y.cpu(), label="True", linewidth=1.0, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

        for layer in models["GAL"].model:
            if isinstance(layer, GAL):
                plt.plot(x.cpu(), layer(x).cpu(), linewidth=1.0)
                plt.grid()
                plt.show()
