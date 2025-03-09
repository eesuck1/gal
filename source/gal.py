from typing import Literal

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# Generalized Adaptive Linear Activation
class GAL(nn.Module):
    def __init__(self, borders: int = 1, device: torch.device = torch.device("cpu"),
                 k_initialization: Literal["ones", "randn"] = "randn", p_initialization: Literal["linspace", "logspace"] = "linspace",
                 trainable_p: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._device = device
        self._n = borders
        self._m = borders + 1

        p_t, p_b = self._create_borders(p_initialization)

        if trainable_p:
            self._p_t = nn.Parameter(p_t)
        else:
            self.register_buffer("_p_t", p_t)

        self.register_buffer("_p_b", p_b)

        self._k = nn.Parameter(self._create_k(k_initialization).to(device))
        self._b_g = nn.Parameter(torch.randn(1))

        k_l_0, k_l_1, k_r_0, k_r_1 = self._create_k_matrix()

        self.register_buffer("_b", torch.zeros((self._n * 2 + 2, 1)))
        self.register_buffer("_k_l_0", k_l_0)
        self.register_buffer("_k_l_1", k_l_1)
        self.register_buffer("_k_r_0", k_r_0)
        self.register_buffer("_k_r_1", k_r_1)

    def _create_borders(self, p_initialization: str) -> tuple[torch.Tensor, torch.Tensor]:
        match p_initialization:
            case "linspace":
                p_l = -torch.linspace(start=1.0 / self._n, end=self._n, steps=self._n).view(-1, 1)
                p_r = torch.linspace(start=1.0 / self._n, end=self._n, steps=self._n).view(-1, 1)

            case "logspace":
                p_l = -torch.logspace(start=-1.0, end=6.0, base=2.0, steps=self._n).view(-1, 1)
                p_r = torch.logspace(start=-1.0, end=6.0, base=2.0, steps=self._n).view(-1, 1)
            case _:
                raise ValueError(f"Unknown initialization type: {p_initialization}")

        p_t = torch.cat([torch.flip(p_l, [0]), p_r])
        p_b = torch.cat([torch.flip(p_l, [0]), torch.zeros((1, 1)), p_r])

        return p_t, p_b

    def _create_k(self, k_initialization: str) -> torch.Tensor:
        n = self._n * 2 + 2

        match k_initialization:
            case "ones":
                k = torch.ones((n, 1))
            case "randn":
                k = torch.randn((n, 1))
            case _:
                raise ValueError(f"Unexpected k initialization type: {k_initialization}")

        return k

    def _create_k_matrix(self) -> tuple[torch.Tensor, ...]:
        n = self._n + 1
        m = n - 1
        k = self._k.expand((self._n * 2 + 2, self._n * 2 + 2)).T

        k_m = torch.zeros((self._n * 2 + 1, self._n * 2 + 1))
        k_m[:m, :m] = torch.triu(torch.ones(m, m))
        k_m[m + 1:, m + 1:] = torch.tril(torch.ones(m, m))

        print(k_m)

        k_m = ~k_m.bool().to(self._device)

        k_0 = torch.masked_fill(k[:-1, :-1], k_m, 0.0)
        k_1 = torch.masked_fill(k[1:, 1:], k_m, 0.0)

        return k_0, k_1


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_l = self._k_l_0 - self._k_l_1
        k_r = self._k_r_0 - self._k_r_1

        b = torch.cat([
            k_l @ self._p_t[:self._n],
            torch.zeros((2, 1), device=self._device),
            k_r @ self._p_t[self._n:]
        ]) + self._b_g

        x_shape = x.shape
        x_flat = x.view(-1)
        x_indexes = torch.bucketize(x_flat, self._p_b.view(-1))

        out = x_flat.view(-1, 1) * self._k[x_indexes] + b[x_indexes]
        out = out.view(x_shape)

        return out


if __name__ == '__main__':
    brs = 3
    d = torch.device("cuda")

    in_sample = torch.linspace(-10.0, 10.0, 1000).to(d)
    gal = GAL(brs, k_initialization="randn", device=d)
    gal.to(d)
    output = gal(in_sample)

    inp = in_sample.cpu().detach().numpy()
    output_cpu = output.cpu().detach().numpy()

    sorted_indexes = inp.argsort()

    plt.plot(inp[sorted_indexes], output_cpu[sorted_indexes])
    plt.grid()
    plt.show()

    loss = output.pow(2).sum()
    loss.backward()

    for name, param in gal.named_parameters():
        print(f"Name: {name}, Value: {param.grad}")
