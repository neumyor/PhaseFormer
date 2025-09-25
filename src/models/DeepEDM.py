import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from src.models.pl_bases.default_module import DefaultPLModule


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        #self.pe.pe.data.shape = torch.Size([1, 1, 1000, 32])
        return self.pe[:, :, offset:offset+x.size(2)]


class InputEncoder(nn.Module):
    def __init__(self,
                  mlp_layers,
                  lookback_len,
                  pred_len,
                  latent_channel_dim,
                  in_channels,
                  activation_fn,
                  use_stamp_data=False,
                  stamp_dim=None,
                  dropout=0.0):
        super(InputEncoder, self).__init__()

        self.in_channels = in_channels

        self.out_channels = latent_channel_dim

        self.lookback_len = lookback_len
        self.pred_len = pred_len

        self.use_stamp_data = use_stamp_data
        self.stamp_dim = stamp_dim

        if use_stamp_data:
            in_channels += stamp_dim 

        mlp_block = []
        for i in range(mlp_layers):
            if i == 0:
                in_features = self.lookback_len
            else:
                in_features = self.pred_len
            
            out_features = self.pred_len
            
            mlp_block.append(nn.Linear(in_features=in_features, out_features=out_features))
            
            if i < mlp_layers-1:
                mlp_block.append(nn.Dropout(dropout))
                mlp_block.append(activation_fn)

        self.mlp_projection = nn.Sequential(*mlp_block)


    def forward(self, x, stamp=None):

        # x.shape -> B, D, T
        B, D, T = x.size()
        if self.use_stamp_data:
            x = torch.cat([x, stamp], dim=1)
               
        # B, D, T -> B, D, T'
        skip_focal_pts = self.mlp_projection(x)
        
        mlp_edm_focal_pts = skip_focal_pts
        
        return x, mlp_edm_focal_pts, skip_focal_pts


class EDMBlock(nn.Module):
    def __init__(
        self,
        lookback_len,
        out_pred_len,
        delay,
        time_delay_stride,
        layer_norm,
        latent_channel_dim,
        method,
        theta,
        add_pe,
        dropout,
        activation_fn,
        dist_projection_dim=64,
        n_proj_layers=1,
    ):
        super().__init__()

        self.lookback_len = lookback_len
        self.out_pred_len = out_pred_len

        self.delay = delay
        self.time_delay_stride = time_delay_stride

        unfolded_len = ((lookback_len + out_pred_len) // time_delay_stride) - delay + 1
        self.unfolded_lookback_len = int(
            (lookback_len / (lookback_len + out_pred_len)) * unfolded_len
        )
        self.unfolded_pred_len = unfolded_len - self.unfolded_lookback_len
        

        self.latent_channel_dim = latent_channel_dim
        self.layer_norm = layer_norm
        self.method = method

        self.theta = theta
        self.activation_fn = activation_fn

        projection = []
        for i in range(n_proj_layers):
            if i == 0:
                projection.append(nn.Linear(self.delay, dist_projection_dim))
            else:
                projection.append(nn.Linear(dist_projection_dim, dist_projection_dim))
                projection.append(nn.Dropout(dropout))
                projection.append(self.activation_fn)

        self.projection = nn.Sequential(*projection)

        if add_pe:
            self.pe = LearnablePositionalEmbedding(
                d_model=dist_projection_dim,
                max_len=max(1100, self.lookback_len + self.out_pred_len),
            )
        else:
            self.pe = None

        self.ln1 = nn.LayerNorm(dist_projection_dim) if layer_norm else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.undelay = nn.Sequential(
            nn.Linear(self.delay * self.unfolded_pred_len, out_pred_len),
            nn.Dropout(dropout),
            self.activation_fn,
            nn.Linear(out_pred_len, out_pred_len),
        )

    def _weight_and_topk(self, td_history_windows, td_next_windows, focal_td_windows_all):
        """
        td_history_windows.shape = [B, D, T_hist, delay]
        td_next_windows.shape   = [B, D, T_hist, delay]  # one-step-shifted windows aligned with history
        focal_td_windows_all    = [B, D, F, delay]       # focal windows for prediction + one extra window
        """
        B, D, F, delay = focal_td_windows_all.size()
        _, _, T, _ = td_history_windows.size()

        k_proj = self.projection(td_history_windows)
        q_proj = self.projection(focal_td_windows_all[:, :, :-1, :])

        if (self.pe is not None) and (k_proj.shape[-1] % 2 == 0):
            k_proj = k_proj + self.pe(k_proj)
            q_proj = q_proj + self.pe(q_proj, offset=k_proj.size(-2))

        k_proj = self.ln1(k_proj)
        q_proj = self.ln1(q_proj)

        # Manual scaled dot-product attention to support V with different embed dim (delay)
        # q_proj: [B, D, F-1, E], k_proj: [B, D, T, E], b (values): [B, D, T, delay]
        scale = (1.0 / math.sqrt(k_proj.size(-1))) * self.theta
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * scale  # [B, D, F-1, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=0.1, training=self.training)
        predictions = torch.matmul(attn_weights, td_next_windows)  # [B, D, F-1, delay]

        return None, predictions, None

    def _edm_forward(self, td_history_windows, td_next_windows, focal_td_windows_all):
        td_history_windows, predictions, W = self._weight_and_topk(
            td_history_windows, td_next_windows, focal_td_windows_all
        )

        if td_history_windows is None and W is None:
            # When we use scaled dot product attention or sparse topk, we get the predictions directly
            return predictions, None
        else:
            raise NotImplementedError(
                "Only scaled dot product attention is implemented for EDM."
            )

    def forward(self, lookback_series, focal_point_series):
        # Concatenate lookback and focal points along the time dimension
        concatenated_series = torch.cat([lookback_series, focal_point_series], dim=-1)

        td_windows = concatenated_series.unfold(-1, self.delay, self.time_delay_stride)  # B,D,T -> B,D,T',delay

        focal_td_windows_all = td_windows[:, :, -self.unfolded_pred_len - 1 :, :]
        td_windows = td_windows[:, :, : -self.unfolded_pred_len - 1, :]

        # B,D,T_hist,delay
        td_history_windows = td_windows[:, :, :-1, :]
        td_next_windows = td_windows[:, :, 1:, :]

        # focal_td_windows_all has one extra point at the end
        all_focal_points = focal_td_windows_all

        # predictions shape: [B, D, F-1, delay]
        predictions, sol = self._edm_forward(td_history_windows, td_next_windows, all_focal_points)

        predictions = predictions.reshape(predictions.size(0), predictions.size(1), -1)
        block_pred = self.undelay(predictions)

        return block_pred, sol


activation_fn_map = {
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'silu': nn.SiLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'gelu': nn.GELU()
}

class DeepEDM(DefaultPLModule):
    def __init__(self, configs, *args, **kwargs):
        super().__init__(configs)

        # Map common repo configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, 'enc_in', getattr(configs, 'c_out', 1)))

        # EDM-specific hyperparameters with sensible defaults
        self.n_edm_blocks = int(getattr(configs, 'n_edm_blocks', 2))
        self.delay = int(getattr(configs, 'delay', 6))
        self.time_delay_stride = int(getattr(configs, 'time_delay_stride', 1))
        self.layer_norm = bool(getattr(configs, 'layer_norm', True))
        self.latent_channel_dim = int(getattr(configs, 'latent_channel_dim', getattr(configs, 'd_model', 256)))
        self.method = getattr(configs, 'method', 'smap')
        self.theta = float(getattr(configs, 'theta', 1.0))
        self.add_pe = bool(getattr(configs, 'add_pe', True))
        self.dropout = float(getattr(configs, 'dropout', 0.1))
        self.dist_projection_dim = int(getattr(configs, 'dist_projection_dim', 64))
        self.n_proj_layers = int(getattr(configs, 'n_proj_layers', 1))

        activation_name = getattr(configs, 'activation', 'gelu')
        self.activation_fn = activation_fn_map.get(activation_name, nn.GELU())

        # Encoder to project lookback to focal points and skip connection
        self.encoder = InputEncoder(
            mlp_layers=int(getattr(configs, 'mlp_layers', 1)),
            lookback_len=self.seq_len,
            pred_len=self.pred_len,
            latent_channel_dim=self.latent_channel_dim,
            in_channels=self.enc_in,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
        )

        # Stacked EDM blocks
        edm_blocks = []
        for _ in range(self.n_edm_blocks):
            edm_blocks.append(
                EDMBlock(
                    lookback_len=self.seq_len,
                    out_pred_len=self.pred_len,
                    delay=self.delay,
                    time_delay_stride=self.time_delay_stride,
                    layer_norm=self.layer_norm,
                    latent_channel_dim=self.latent_channel_dim,
                    method=self.method,
                    theta=self.theta,
                    add_pe=self.add_pe,
                    dropout=self.dropout,
                    activation_fn=self.activation_fn,
                    dist_projection_dim=self.dist_projection_dim,
                    n_proj_layers=self.n_proj_layers,
                )
            )
        self.edm_blocks = nn.ModuleList(edm_blocks)
        self.gate_edm = nn.Linear(self.pred_len, 1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        # x_enc: [B, L, D] -> [B, D, L]
        x_per_channel = x_enc.permute(0, 2, 1)

        # Normalize per variable (channel-wise)
        channel_means = x_per_channel.mean(dim=-1, keepdim=True).detach()
        x_normalized = x_per_channel - channel_means
        channel_stds = torch.sqrt(torch.var(x_normalized, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_normalized = x_normalized / channel_stds

        normalized_lookback_series, focal_point_series, skip_connection_focal_points = self.encoder(x_normalized)

        for i in range(len(self.edm_blocks)):
            block_pred, _ = self.edm_blocks[i](normalized_lookback_series, focal_point_series)
            focal_point_series = block_pred

        edm_prediction = block_pred  # [B, D, pred_len]

        edm_gate = self.gate_edm(edm_prediction).sigmoid()  # [B, D, 1]
        fused_prediction = (edm_prediction * edm_gate) + skip_connection_focal_points  # [B, D, pred_len]

        # de-normalize
        fused_prediction = fused_prediction * channel_stds + channel_means

        # [B, D, pred_len] -> [B, pred_len, D]
        fused_prediction = fused_prediction.permute(0, 2, 1)

        return fused_prediction