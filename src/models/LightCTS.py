import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Optional, Any, Union, Callable
from einops import rearrange
from torch import Tensor
from torch.nn import *
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import numpy as np

NUM_VAR = None


class Transformer(nn.Module):
    def __init__(self, d_model=32, n_heads=8, layers=6):
        super(Transformer, self).__init__()

        self.layers = 4
        self.hid_dim = d_model
        self.heads = n_heads

        self.attention_layer = LightformerLayer(
            self.hid_dim, self.heads, self.hid_dim * 4
        )
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = Lightformer(
            self.attention_layer, self.layers, self.attention_norm
        )
        # self.lpos = LearnedPositionalEncoding(self.hid_dim)
        self.lpos = nn.Parameter(
            torch.randn((NUM_VAR, 1, self.hid_dim), dtype=torch.float32)
        )

    def forward(self, input, mask=None):
        x = input.permute(1, 0, 2)
        x = self.lpos + x
        output = self.attention(x, mask)

        return output.permute(1, 0, 2)


class LScaledDotProductAttention(Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1, groups=2):

        super(LScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_k = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_v = nn.Linear(d_model // groups, h * d_v // groups)
        self.fc_o = nn.Linear(h * d_v // groups, d_model // groups)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.fc_q.weight)
        xavier_uniform_(self.fc_k.weight)
        xavier_uniform_(self.fc_v.weight)
        xavier_uniform_(self.fc_o.weight)
        constant_(self.fc_q.bias, 0)
        constant_(self.fc_k.bias, 0)
        constant_(self.fc_v.bias, 0)
        constant_(self.fc_o.bias, 0)

    def forward(
        self, queries, keys, values, attention_mask=None, attention_weights=None
    ):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = (
            self.fc_q(queries.view(b_s, nq, self.groups, -1))
            .view(b_s, nq, self.h, self.d_k)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.fc_k(keys.view(b_s, nk, self.groups, -1))
            .view(b_s, nk, self.h, self.d_k)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.fc_v(values.view(b_s, nk, self.groups, -1))
            .view(b_s, nk, self.h, self.d_v)
            .permute(0, 2, 1, 3)
        )

        att = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = (
            torch.matmul(att, v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b_s, nq, self.h * self.d_v)
        )
        out = self.fc_o(out.view(b_s, nq, self.groups, -1)).view(b_s, nq, -1)
        return out.permute(1, 0, 2)


class LMultiHeadAttention(Module):
    def __init__(self, d_model, h, dropout=0.1, batch_first=False, groups=2):
        super(LMultiHeadAttention, self).__init__()

        self.attention = LScaledDotProductAttention(
            d_model=d_model,
            groups=groups,
            d_k=d_model // h,
            d_v=d_model // h,
            h=h,
            dropout=dropout,
        )

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        attention_weights=None,
    ):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out


class Lightformer(Module):

    __constants__ = ["norm"]

    def __init__(self, attention_layer, num_layers, norm=None):
        super(Lightformer, self).__init__()
        self.layers = _get_clones(attention_layer, num_layers - 1)
        self.layers.append(attention_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for i, mod in enumerate(self.layers):
            if i % 2 == 0:
                output = mod(output)
            else:
                output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LightformerLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
    ):
        super(LightformerLayer, self).__init__()
        self.self_attn = LMultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward // 2, d_model // 2)  ###

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x))).view(b, l, 2, d * 4 // 2)
        )
        x = x.view(b, l, d)
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class MyTransformer(nn.Module):
    def __init__(self, hid_dim, layers, heads=8):
        super().__init__()
        self.heads = heads
        self.layers = layers
        self.hid_dim = hid_dim
        self.trans = Transformer(hid_dim, heads, layers)

    def forward(self, x, mask=None):
        x = self.trans(x, mask)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


from easydict import EasyDict as edict

import csv


def get_adj_matrix(
    distance_df_filename, num_of_vertices, type_="connectivity", id_filename=None
):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, "r") as f:
            id_dict = {
                int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
            }
        with open(distance_df_filename, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == "connectivity":
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == "distance":
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")

    return torch.tensor(A)


class LightCTS(nn.Module):
    def __init__(self, configs: edict):
        super(LightCTS, self).__init__()

        self.dropout = configs.dropout
        self.adj_csv_path = configs.adj_csv_path
        self.in_dim = 1
        self.out_dim = configs.pred_len
        self.hid_dim = configs.d_model
        self.layers = configs.e_layers
        self.group = 4
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        global NUM_VAR
        NUM_VAR = configs.enc_in

        # if self.adj_csv_path is not None:
        #     self.supports = get_adj_matrix(distance_df_filename=self.adj_csv_path, num_of_vertices=self.enc_in)
        # else:
        #     self.supports = 1 - torch.eye(self.enc_in, self.enc_in)

        self.start_conv = Conv2d(
            in_channels=self.in_dim, out_channels=self.hid_dim, kernel_size=(1, 1)
        )
        self.cnn_layers = 8
        self.group = 4
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        D = [1, 2, 4, 8, 16, 32, 48, 64]
        receptive_field = 1

        for i in range(self.cnn_layers):
            self.filter_convs.append(
                Conv2d(
                    self.hid_dim, self.hid_dim, (1, 2), dilation=D[i], groups=self.group
                )
            )
            self.gate_convs.append(
                Conv2d(
                    self.hid_dim, self.hid_dim, (1, 2), dilation=D[i], groups=self.group
                )
            )
            receptive_field += D[i]

        self.receptive_field = receptive_field
        depth = list(range(self.cnn_layers - 1))
        self.bn = ModuleList([BatchNorm2d(self.hid_dim) for _ in depth])

        self.end_conv1 = nn.Linear(self.hid_dim, self.hid_dim * 4)
        self.end_conv2 = nn.Linear(self.hid_dim * 4, self.out_dim)

        self.network = MyTransformer(self.hid_dim, layers=self.layers, heads=8)

        # mask0 = self.supports[0].detach()
        # mask1 = self.supports[1].detach()
        # mask = mask0 + mask1
        # self.mask = mask == 0
        # self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.se = SELayer(self.hid_dim)

    def forward(self, x_enc, *args, **kwargs):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.unsqueeze(-1)
        input = rearrange(x_enc, "b l n z -> b z n l")
        # input: [bsz, 1, n_var, seq_len]

        in_len = input.size(3)
        if in_len < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))

        x = self.start_conv(input)
        skip = 0

        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            if self.group != 1:
                x = channel_shuffle(x, self.group)

            try:
                skip += x[:, :, :, -1:]
            except:
                skip = 0

            if i == self.cnn_layers - 1:
                break

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = torch.squeeze(skip, dim=-1)
        x = self.se(x)
        x = x.transpose(1, 2)
        x_residual = x
        x = self.network(x)
        x += x_residual
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        x = x.transpose(1, 2).unsqueeze(-1)

        dec_out = x.squeeze(-1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
