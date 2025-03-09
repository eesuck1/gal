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

        p_l, p_r = self._create_borders(p_initialization)

        if trainable_p:
            self._p_l, self._p_r = nn.Parameter(p_l), nn.Parameter(p_r)
        else:
            self.register_buffer("_p_l", p_l)
            self.register_buffer("_p_r", p_r)

        self._k_l, self._k_r = self._create_k(k_initialization)

        self._b_g = nn.Parameter(torch.randn(1))

        self.register_buffer("_k_l_matrix", self._k_l.expand((self._n + 1, self._n + 1)).T)
        self.register_buffer("_k_r_matrix", self._k_r.expand((self._n + 1, self._n + 1)).T)

        self.register_buffer("_m", self._create_mask())
        self.register_buffer("_inf", torch.tensor([[float("inf")]]))
        self.register_buffer("_zero", torch.tensor([[0.0]]))

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

        return p_l, p_r

    def _create_mask(self) -> torch.Tensor:
        n = self._n + 1
        m = torch.tril(torch.ones((n, n)))

        return ~m.bool()

    def _create_k(self, k_initialization: str) -> tuple[torch.Tensor, torch.Tensor]:
        n = self._n + 1

        match k_initialization:
            case "ones":
                k_l = torch.ones((n, 1))
                k_r = torch.ones((n, 1))
            case "randn":
                k_l = torch.randn((n, 1))
                k_r = torch.randn((n, 1))
            case _:
                raise ValueError(f"Unexpected k initialization type: {k_initialization}")

        return nn.Parameter(k_l.to(self._device)), nn.Parameter(k_r.to(self._device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_l = torch.cat([self._zero, self._p_l])
        p_l_flip = torch.flip(p_l, [0])
        p = torch.cat([-self._inf, p_l_flip, self._p_r, self._inf])

        k_l = self._k_l_matrix.masked_fill(self._m, 0.0)
        k_r = self._k_r_matrix.masked_fill(self._m, 0.0)
        k = torch.cat([torch.flip(self._k_l, [0]), self._k_r])

        b_l = (k_l[:-1, :-1] - k_l[1:, 1:]) @ self._p_l
        b_l = torch.cat([self._zero, b_l])

        b_r = (k_r[:-1, :-1] - k_r[1:, 1:]) @ self._p_r
        b_r = torch.cat([self._zero, b_r])

        b = torch.cat([torch.flip(b_l, [0]), b_r]) + self._b_g

        x_shape = x.shape
        x_flat = x.view(-1)

        indexes = torch.bucketize(x_flat, p.flatten()) - 1

        out = x_flat.view(-1, 1) * k[indexes] + b[indexes]
        out = out.view(x_shape)

        return out


if __name__ == '__main__':
    brs = 3
    d = torch.device("cuda")

    in_sample = torch.randn(1000).to(d)
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
