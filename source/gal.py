import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# Generalized Adaptive Linear Activation
class GAL(nn.Module):
    def __init__(self, borders: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._left = nn.Parameter(-torch.logspace(start=-borders / 2.0, end=borders / 2.0, steps=borders, base=2.0))
        self._right = nn.Parameter(torch.logspace(start=-borders / 2.0, end=borders / 2.0, steps=borders, base=2.0))

        self._k_l = nn.Parameter(torch.randn(borders + 1) * 0.1)
        self._k_r = nn.Parameter(torch.randn(borders + 1) * 0.1)

        self._b_g = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, device=x.device)
        zero = torch.tensor([0.0], requires_grad=False, device=x.device)
        inf = torch.tensor([float("inf")], requires_grad=False, device=x.device)

        y = self._b_g

        for k_0, (p_0, p_1) in zip(self._k_r,
                                   zip(torch.cat([zero, self._right]), torch.cat([self._right, inf]))):
            mask = (x > p_0) & (x <= p_1)
            b = y - k_0 * p_0
            y = b + k_0 * p_1

            out[mask] = x[mask] * k_0 + b

        y = self._b_g

        for k_0, (p_0, p_1) in zip(self._k_l,
                                   zip(torch.cat([zero, self._left]), torch.cat([self._left, -inf]))):
            mask = (x > p_1) & (x <= p_0)
            b = y - k_0 * p_0
            y = b + k_0 * p_1

            out[mask] = x[mask] * k_0 + b

        return out


if __name__ == '__main__':
    with torch.no_grad():
        brs = 10
        p = torch.logspace(start=-brs / 2.0, end=brs / 2.0, steps=brs, base=2.0)

        plt.step(torch.arange(brs), p)
        plt.grid()
        plt.show()

        in_sample = torch.linspace(-10, 10, 1000)
        gal = GAL(brs)

        plt.plot(in_sample, gal(in_sample))
        plt.grid()
        plt.show()
