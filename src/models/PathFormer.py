import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from .layers.PathFormer_AMS import AMS
from .layers.PathFormer_Layer import WeightGenerator, CustomLinear
from .layers.PathFormer_RevIN import RevIN
from .pl_bases.default_module import DefaultPLModule
from functools import reduce
from operator import mul


class PathFormer(DefaultPLModule):
    def __init__(self, configs):
        super(PathFormer, self).__init__(configs)
        
        # PathFormer specific parameters
        self.layer_nums = getattr(configs, 'layer_nums', 3)  # 设置pathway的层数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.k = getattr(configs, 'k', 2)
        self.num_experts_list = getattr(configs, 'num_experts_list', [4, 4, 4])
        self.patch_size_list = getattr(configs, 'patch_size_list', [[16,12,8,32], [12,8,6,4], [8,6,4,2]])
        self.d_model = getattr(configs, 'd_model', 16)
        self.d_ff = getattr(configs, 'd_ff', 64)
        self.residual_connection = getattr(configs, 'residual_connection', 0)
        self.batch_norm = getattr(configs, 'batch_norm', 0)
        self.revin = getattr(configs, 'revin', 1)
        
        if self.revin:
            self.revin_layer = RevIN(num_features=self.enc_in, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], None, k=self.k,
                    num_nodes=self.enc_in, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, 
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pred_len)
        )
    


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        # PathFormer only uses x_enc input
        x = x_enc  # [B, L, D]
        
        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        # Transform to [B, L, D, 1] for PathFormer processing
        out = self.start_fc(x.unsqueeze(-1))  # [B, L, D, d_model]

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        # Reshape output: [B, L, D, d_model] -> [B, D, L*d_model] -> [B, D, pred_len]
        out = out.permute(0,2,1,3).reshape(batch_size, self.enc_in, -1)
        out = self.projections(out).transpose(2, 1)  # [B, pred_len, D]

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        # Store balance loss for training
        self.balance_loss = balance_loss
        
        return out

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        outputs = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # Get final predictions
        outputs = outputs[:, -self.pred_len :, :]
        batch_y = batch_y[:, -self.pred_len :, :]

        if hasattr(self, 'target_var_index') and self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion("mae")
        
        # Main prediction loss
        main_loss = criterion(outputs, batch_y)
        
        # Add balance loss from PathFormer
        total_loss = main_loss + self.balance_loss
        
        # Logging
        self.log("train_loss_main", main_loss, on_epoch=True)
        self.log("train_loss_balance", self.balance_loss, on_epoch=True)
        self.log("train_loss", total_loss, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        outputs = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        outputs = outputs[:, -self.pred_len :, :]
        batch_y = batch_y[:, -self.pred_len :, :]

        if hasattr(self, 'target_var_index') and self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion("mae")
        loss = criterion(outputs, batch_y)
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        outputs = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        outputs = outputs[:, -self.pred_len :, :]
        batch_y = batch_y[:, -self.pred_len :, :]

        if hasattr(self, 'target_var_index') and self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        pred = outputs.detach()
        true = batch_y.detach()

        from src.utils.metrics import metric

        loss = metric(pred, true)
        self.log_dict({f"test_{k}": v for k, v in loss.items()}, on_epoch=True)

        return loss
