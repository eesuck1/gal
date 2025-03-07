import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# Generalized Adaptive Linear Activation
class GAL(nn.Module):
    def __init__(self, borders: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._n = borders
        self._p_l, self._p_r = self._create_borders()
        self._k_l, self._k_r = self._create_k()

        self._b_g = nn.Parameter(torch.randn(1) * 0.1)

        self.register_buffer("_m", self._create_mask())
        self.register_buffer("_k_l_matrix", self._k_l.expand((self._n + 1, self._n + 1)).T)
        self.register_buffer("_k_r_matrix", self._k_r.expand((self._n + 1, self._n + 1)).T)

    def _create_borders(self) -> tuple[torch.Tensor, torch.Tensor]:
        p_l = -torch.logspace(start=-self._n / 2.0, end=self._n / 2.0, steps=self._n, base=2.0).view(-1, 1)
        p_r = torch.logspace(start=-self._n / 2.0, end=self._n / 2.0, steps=self._n, base=2.0).view(-1, 1)
        p_0 = torch.zeros((1, 1))

        return nn.Parameter(torch.cat([p_0, p_l])), nn.Parameter(torch.cat([p_0, p_r]))

    def _create_mask(self) -> torch.Tensor:
        n = self._n + 1
        m = torch.tril(torch.ones((n, n)))

        return ~m.bool()

    def _create_k(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self._n + 1

        k_l = torch.randn((n, 1)) * 0.1
        k_r = torch.randn((n, 1)) * 0.1

        return nn.Parameter(k_l), nn.Parameter(k_r)

    @staticmethod
    def forward_side(x: torch.Tensor, out: torch.Tensor, k: torch.Tensor, p: torch.Tensor, is_left: bool) -> torch.Tensor:
        b = (k[:-1, :-1] - k[1:, 1:]) @ p[1:]
        b = torch.cat([torch.zeros(1, device=x.device, requires_grad=False), b.view(-1)])

        p_values = zip(p[1:], p[:-1]) if is_left else zip(p[:-1], p[1:])

        for i, (p_0, p_1) in enumerate(p_values):
            mask = (x > p_0) & (x <= p_1)
            out[mask] = x[mask] * k[-1, i] + b[i]

        mask = x < p[-1] if is_left else x > p[-1]
        out[mask] = x[mask] * k[-1, -1] + b[-1]

        return out


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_l = self._k_l_matrix.masked_fill(self._m, 0.0)
        k_r = self._k_r_matrix.masked_fill(self._m, 0.0)

        out = torch.zeros_like(x, device=x.device)

        self.forward_side(x, out, k_l, self._p_l, True)
        self.forward_side(x, out, k_r, self._p_r, False)

        out += self._b_g

        return out


if __name__ == '__main__':
    brs = 3
    p1 = torch.logspace(start=-brs / 2.0, end=brs / 2.0, steps=brs, base=2.0)

    in_sample = torch.linspace(-10, 10, 3000)
    gal = GAL(brs)
    output = gal(in_sample)

    plt.plot(in_sample.detach().numpy(), output.detach().numpy())
    plt.grid()
    plt.show()

    loss = output.pow(2).sum()
    loss.backward()

    for p in gal.parameters():
        print(p.grad)
