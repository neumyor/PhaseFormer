import torch
from torch import nn
import torch.nn.functional as F
from src.models.layers.Transformer_EncDec import Encoder
from src.models.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)
from src.models.layers.Embed import PatchEmbedding
from src.models.pl_bases.default_module import DefaultPLModule


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TransBatchNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
        )

    def forward(self, x):
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        skip_attn=False,
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.skip_attn = skip_attn

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.skip_attn:
            new_x = x
        else:
            new_x, attn = self.attention(
                x, x, x, attn_mask=attn_mask, tau=tau, delta=delta
            )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class PatchTST(DefaultPLModule):
    def __init__(self, configs, *args, **kwargs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__(configs)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = configs.stride
        self.use_norm = configs.use_norm
        self.skip_attn = configs.get("skip_attn", False)

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            configs.patch_len,
            configs.stride,
            padding,
            configs.emb_dropout,
            configs.emb_bias,
            configs.pos_max_len,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    skip_attn=self.skip_attn,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.GroupNorm(1, configs.d_model), Transpose(1, 2)
            ),
        )

        # Prediction Head
        self.head_nf = configs.d_model * int(
            (configs.seq_len - configs.patch_len) / configs.stride + 2
        )

        self.head = FlattenHead(
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forward(self, x_enc, *args, **kwargs):
        # x_enc: [B, L, D]
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        self.attns = attns
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
        return dec_out
