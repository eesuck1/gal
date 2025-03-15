from typing import Literal

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import torchviz


# Generalized Adaptive Linear Activation
class GAL(nn.Module):
    def __init__(self, borders: int = 1, k_initialization: Literal["ones", "randn", "leaky_relu"] = "randn",
                 p_initialization: Literal["linspace", "logspace", "cumulative_uniform"] = "linspace",
                 trainable_p: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._n = borders

        p = self._create_borders(p_initialization)

        if trainable_p:
            self._p = nn.Parameter(p)
        else:
            self.register_buffer("_p", p)

        self._k = nn.Parameter(self._create_k(k_initialization))
        self._b_g = nn.Parameter(torch.randn(1))


    def _create_borders(self, p_initialization: str) -> torch.Tensor:
        match p_initialization:
            case "linspace":
                p_l = -torch.linspace(start=1.0 / self._n, end=self._n, steps=self._n)
                p_r = torch.linspace(start=1.0 / self._n, end=self._n, steps=self._n)
            case "logspace":
                p_l = -torch.logspace(start=-1.0, end=6.0, base=2.0, steps=self._n)
                p_r = torch.logspace(start=-1.0, end=6.0, base=2.0, steps=self._n)
            case "cumulative_uniform":
                p_l = -torch.rand(self._n).clamp(min=0.1).cumsum(dim=0)
                p_r = torch.rand(self._n).clamp(min=0.1).cumsum(dim=0)
            case _:
                raise ValueError(f"Unknown initialization type: {p_initialization}")

        p_l = torch.flip(p_l, [0])
        p = torch.cat([p_l, torch.zeros(1), p_r])

        return p

    def _create_k(self, k_initialization: str) -> torch.Tensor:
        n = self._n + 1

        match k_initialization:
            case "ones":
                k_l = torch.ones(n)
                k_r = torch.ones(n)
            case "randn":
                k_l = torch.randn(n)
                k_r = torch.randn(n)
            case "leaky_relu":
                k_l = torch.ones(n) * 1e-2
                k_r = torch.ones(n)
            case _:
                raise ValueError(f"Unexpected k initialization type: {k_initialization}")

        k_l = torch.flip(k_l, [0])
        k = torch.cat([k_l, k_r])

        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (self._k[0] + self._k[-1]) / 2.0
        c = (self._k[1:] - self._k[:-1]) / 2.0

        out = torch.sum(torch.abs(torch.sub(x.unsqueeze(-1), self._p)) * c, dim=-1)
        out = a * x + out + self._b_g

        return out


if __name__ == '__main__':
    brs = 3
    d = torch.device("cuda")

    in_sample = torch.randn(1000).to(d)
    gal = GAL(brs, k_initialization="leaky_relu")
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

    # graph = torchviz.make_dot(loss, params=dict(gal.named_parameters()), show_attrs=True, show_saved=True)
    # graph.render(f"../assets/gal", format="png")

    for name, param in gal.named_parameters():
        print(f"Name: {name}, Value: {param.grad}")
