import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT4TS(nn.Module):
    def __init__(
        self,
        patch_len,
        stride,
        input_len,
        output_len,
        num_encoder_layers,
        *args,
        **kwargs
    ):
        super(GPT4TS, self).__init__()
        self.is_gpt = True
        self.pretrain = True
        self.freeze = True

        self.patch_size = patch_len
        self.stride = stride
        self.patch_num = (input_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if self.is_gpt:
            if self.pretrain:
                self.gpt2 = GPT2Model.from_pretrained(
                    "gpt2",
                    output_attentions=True,
                    output_hidden_states=True,
                    cache_dir="./resources/hf_weights",
                    local_files_only=True,
                )  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:num_encoder_layers]
            print("gpt2 = {}".format(self.gpt2))

        token_dim = 768
        self.in_layer = nn.Linear(patch_len, token_dim)
        self.out_layer = nn.Linear(token_dim * self.patch_num, output_len)

        if self.freeze and self.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.train()

        self.cnt = 0

    def forward(self, x_enc, *args, **kwargs):
        """
        x: tensor (b, n, l)
        """
        x = x_enc
        x = rearrange(x, "b l n -> b n l")

        B, M, L = x.shape
        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x /= stdev
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b m n p -> (b m) n p")

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, "(b m) l -> b m l", b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        outputs = rearrange(outputs, "b n l -> b l n")

        return outputs
