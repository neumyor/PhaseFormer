import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphWavenet(nn.Module):
    def __init__(self, configs):
        super(GraphWavenet, self).__init__()
        self.configs = configs
        self.dropout = configs.dropout
        # self.blocks = int(configs.seq_len / 3)
        self.blocks = configs.blocks
        self.layers = configs.layers
        self.gcn_bool = configs.gcn_bool
        self.addaptadj = configs.addaptadj

        in_dim = 1
        out_dim = configs.pred_len

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=configs.residual_channels, kernel_size=(1, 1)
        )
        self.supports = configs.supports

        receptive_field = 1

        self.supports_len = 0
        if configs.supports is not None:
            self.supports_len += len(configs.supports)

        if configs.gcn_bool and configs.addaptadj:
            if configs.aptinit is None:
                if configs.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(configs.num_nodes, 10), requires_grad=True
                )
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, configs.num_nodes), requires_grad=True
                )
                self.supports_len += 1
            else:
                if configs.supports is None:
                    self.supports = []
                m, p, n = torch.svd(configs.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.configs.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=configs.residual_channels,
                        out_channels=configs.dilation_channels,
                        kernel_size=(1, self.configs.kernel_size),
                        dilation=new_dilation,
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=configs.residual_channels,
                        out_channels=configs.dilation_channels,
                        kernel_size=(1, self.configs.kernel_size),
                        dilation=new_dilation,
                    )
                )

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=configs.dilation_channels,
                        out_channels=configs.skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(configs.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            configs.dilation_channels,
                            configs.residual_channels,
                            dropout=self.dropout,
                            support_len=self.supports_len,
                        )
                    )

        self.end_conv_1 = nn.Conv2d(
            in_channels=configs.skip_channels,
            out_channels=configs.end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=configs.end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.receptive_field = receptive_field

        self.bn[-1].requires_grad_(False)

    def forward(self, x_enc, *args, **kwargs):
        # x [B, L, D]
        # input [B, 1, D, L]
        input = x_enc.permute(0, 2, 1).unsqueeze(1)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x.squeeze(-1)