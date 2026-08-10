import torch.nn as nn
from einops import rearrange, repeat


class InspectLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Rearrange(nn.Module):
    def __init__(self, pattern: str, **kwargs) -> None:
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        x = rearrange(x, self.pattern, **self.kwargs)
        return x


class SkipConnectionWrapper(nn.Module):
    def __init__(self, wrapped_module) -> None:
        super().__init__()
        self.wrapped_module = wrapped_module

    def forward(self, x):
        residual = x
        x = self.wrapped_module(x)
        return x + residual


class BypassWrapper(nn.Module):
    def __init__(self, wrapped_module, bypass_layer):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.bypass_layer = bypass_layer

    def forward(self, x):
        gate = self.bypass_layer(x)
        out = self.wrapped_module(x)
        return gate * out
