"""Linear probes with several softmax heads sharing one input.

A board probe has 64 heads (squares) of 13 classes; a side-to-move probe has
one head of two classes. Logits are ``(x - mean) @ weight + bias``, so a class
direction is a column of ``weight`` and does not depend on the mean.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Probe:
    mean: torch.Tensor      # [width]
    weight: torch.Tensor    # [width, heads * classes]
    bias: torch.Tensor      # [heads * classes]
    heads: int
    classes: int

    def logits(self, x):
        x = x.to(self.weight.device, torch.float32)
        return ((x - self.mean) @ self.weight + self.bias).view(len(x), self.heads, self.classes)

    def predict(self, x, batch=8192):
        return torch.cat([self.logits(x[i:i + batch]).argmax(-1).cpu() for i in range(0, len(x), batch)])

    def direction(self, head, source, target):
        """Residual direction that raises ``target`` over ``source`` for one head."""
        w = self.weight.view(-1, self.heads, self.classes)[:, head]
        return w[:, target] - w[:, source]

    def state(self):
        return {key: getattr(self, key) if key in ('heads', 'classes') else getattr(self, key).cpu()
                for key in ('mean', 'weight', 'bias', 'heads', 'classes')}

    @classmethod
    def from_state(cls, state, device='cpu'):
        return cls(**{key: value.to(device) if torch.is_tensor(value) else value
                      for key, value in state.items()})


def fit(x, y, classes, *, epochs=6, batch=2048, lr=1e-3, weight_decay=1e-4, device='cpu', seed=0):
    """Fit a probe to ``x`` [n, width] and integer labels ``y`` [n, heads]."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(x).to(device, torch.float32)
    y = torch.as_tensor(np.asarray(y, dtype=np.int64)).to(device)
    heads = y.shape[1]
    mean = x.mean(0)
    linear = torch.nn.Linear(x.shape[1], heads * classes).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * ((len(x) + batch - 1) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator).to(device)
        for start in range(0, len(x), batch):
            rows = order[start:start + batch]
            logits = linear(x[rows] - mean).view(len(rows), heads, classes)
            loss = F.cross_entropy(logits.reshape(-1, classes), y[rows].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
    return Probe(mean.detach(), linear.weight.detach().T.contiguous(), linear.bias.detach(), heads, classes)
