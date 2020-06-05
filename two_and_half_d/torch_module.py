import torch
from torch import nn

from dpipe.torch import moveaxis, sequence_to_var, optimizer_step, to_np
from dpipe.itertools import squeeze_first


class Quasi3DWrapper(nn.Module):
    def __init__(self, network: nn.Module, axis=-1):
        super().__init__()
        self.network = network
        self.axis = axis

    def forward(self, xs):
        bs, n_slices = len(xs), xs.shape[self.axis]
        # join self.axis with batch dim
        xs = moveaxis(xs, self.axis, 1)
        xs = xs.reshape(-1, *xs.shape[2:])

        xs = self.network(xs)
        # handling multiple outputs
        if isinstance(xs, torch.Tensor):
            xs = xs,

        # move self.axis back
        results = []
        for x in xs:
            x = x.reshape(bs, n_slices, *x.shape[1:])
            x = moveaxis(x, 1, self.axis)
            results.append(x)

        return squeeze_first(results)


class CRFWrapper(nn.Module):
    def __init__(self, network: nn.Module, crf: nn.Module):
        super().__init__()
        self.network = network
        self.crf = crf

    def forward(self, x, spatial_spacings=None):
        return self.crf(self.network(x), spatial_spacings)


def crf_train_step(x, spacings, target, architecture: CRFWrapper, criterion, optimizer, lr, with_crf: bool):
    architecture.train()

    x, target = sequence_to_var(x, target, device=architecture)

    output = architecture(x, spacings) if with_crf else architecture.network(x)
    loss = criterion(output, target)

    optimizer_step(optimizer, loss, lr=lr)
    return to_np(loss)
