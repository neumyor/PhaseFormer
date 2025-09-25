import torch
import torch.nn as nn


class LoraLinearWrapper(nn.Module):
    def __init__(self, in_dim, out_dim, rank, dropout, scale, wrapped_module):
        super(LoraLinearWrapper, self).__init__()
        self.lora_down = nn.Linear(in_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_dim, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        torch.nn.init.zeros_(self.lora_up.weight)
        # torch.nn.init.zeros_(self.lora_up.bias)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight)
        # torch.nn.init.kaiming_uniform_(self.lora_down.bias)

        self.wrapped_module = wrapped_module
        self.scale = scale

    def forward(self, x):
        result = self.wrapped_module(x)
        lora_result = self.lora_up(self.lora_dropout(self.lora_down(x)))
        return result + lora_result * self.scale
