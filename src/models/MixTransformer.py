import torch
import torch.nn as nn
import torch.nn.functional as F
from .iTransformer import iTransformer
from .PatchTST import PatchTST
from src.models.pl_bases.default_module import DefaultPLModule
from src.utils.metrics import metric
import numpy as np
import copy
import math


class MixTransformer(DefaultPLModule):
    def __init__(self, configs):
        super(MixTransformer, self).__init__(configs)
        itrans_configs = copy.deepcopy(configs)
        itrans_configs.pred_len = itrans_configs.pred_len + 1  # 最后一位标记为输出权重
        itrans_configs.use_norm = False
        print("iTransformer", itrans_configs.seq_len, itrans_configs.pred_len)
        self.itransformer = iTransformer(itrans_configs)

        patchtst_configs = copy.deepcopy(configs)
        patchtst_configs.seq_len = configs.pred_len + configs.seq_len
        patchtst_configs.e_layers = 2
        patchtst_configs.use_norm = False
        patchtst_configs.pos_max_len = patchtst_configs.seq_len
        self.pred_len = configs.pred_len
        print("PatchTST", patchtst_configs.seq_len, patchtst_configs.pred_len)
        self.patchtst = PatchTST(patchtst_configs)

        if configs.finetuning:
            # iTransformer
            configs_folder = f"./log/training_results/iTransformer/{configs.dataset_args.dataset}-{itrans_configs.seq_len}-{itrans_configs.pred_len-1}/checkpoints/"
            import os

            # 获取ckpt文件列表
            ckpt_files = os.listdir(configs_folder)
            ckpt_files = [f for f in ckpt_files if f.endswith(".ckpt")]
            best_ckpt = ckpt_files[0]
            original_ckpt_path = os.path.join(configs_folder, best_ckpt)

            # 加载原始checkpoint
            checkpoint = torch.load(original_ckpt_path)

            # 过滤掉projector模块的参数
            filtered_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                # 排除所有以projector.开头的参数
                if not key.startswith("projector."):
                    filtered_state_dict[key] = value

            # 更新checkpoint中的状态字典
            checkpoint["state_dict"] = filtered_state_dict

            # 保存处理后的临时checkpoint
            temp_ckpt_path = os.path.join(configs_folder, "_temp.ckpt")
            torch.save(checkpoint, temp_ckpt_path)

            # 从临时checkpoint加载模型
            self.itransformer = iTransformer.load_from_checkpoint(
                temp_ckpt_path,
                configs=itrans_configs,
                strict=False,  # 使用strict=False以允许缺失的projector参数
            )

            # 可选：删除临时文件
            os.remove(temp_ckpt_path)
            print(f"Loaded iTransformer from {configs_folder + best_ckpt}")

            # 冻结iTransformer的参数，但是最后投影头不冻结
            for name, param in self.itransformer.named_parameters():
                if "projector" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # PatchTST
            configs_folder = f"./log/training_results/PatchTST/{configs.dataset_args.dataset}-{patchtst_configs.seq_len - configs.pred_len}-{patchtst_configs.pred_len}/checkpoints/"
            # 从configs_folder下面加载ckpt文件，需要用os先确定ckpt文件具体文件名
            import os

            ckpt_files = os.listdir(configs_folder)
            ckpt_files = [f for f in ckpt_files if f.endswith(".ckpt")]
            best_ckpt = ckpt_files[0]
            # self.patchtst = PatchTST.load_from_checkpoint(
            #     configs_folder + best_ckpt, configs=patchtst_configs
            # )
            ckpt = torch.load(configs_folder + best_ckpt, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if (not k.startswith("head.")) and ("position_embedding" not in k)
            }

            self.patchtst.load_state_dict(filtered_state_dict, strict=False)

            print(f"Loaded PatchTST from {configs_folder + best_ckpt}")

            # 冻结PatchTST的参数，但是只要是head.xxx就不冻结
            for name, param in self.patchtst.named_parameters():
                if "head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # 缓存信息以供检查与可视化
            self._temp_gate = None
            self._temp_itrans_output = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc /= stdev

        # Embedding
        dec_out_1 = self.itransformer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out_1, gate = dec_out_1[:, : self.pred_len], dec_out_1[:, -1].unsqueeze(1)
        x_y = torch.cat([x_enc, dec_out_1], dim=1)
        dec_out_2 = self.patchtst(x_y, x_mark_enc, x_dec, x_mark_dec)
        dec_out_2 = (
            torch.sigmoid(gate) * dec_out_1 + (1 - torch.sigmoid(gate)) * dec_out_2
        )

        dec_out_1 = dec_out_1 * (stdev[:, 0, :].unsqueeze(1))
        dec_out_1 = dec_out_1 + (means[:, 0, :].unsqueeze(1))

        if not self.training:
            self._temp_gate = torch.sigmoid(gate)
            self._temp_itrans_output = dec_out_1

        dec_out_2 = dec_out_2 * (stdev[:, 0, :].unsqueeze(1))
        dec_out_2 = dec_out_2 + (means[:, 0, :].unsqueeze(1))

        if self.training:
            return dec_out_1, dec_out_2
        return dec_out_2

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        dec_out_1, dec_out_2 = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        batch_y = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(dec_out_1, batch_y) + criterion(dec_out_2, batch_y)
        self.log("train_loss", loss, on_step=True)
        self.log("d1_loss", criterion(dec_out_1, batch_y), on_step=True)
        self.log("d2_loss", criterion(dec_out_2, batch_y), on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        dec_out = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )
        batch_y = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(dec_out, batch_y)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        dec_out = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        batch_y = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        pred = dec_out.detach()
        true = batch_y.detach()

        results = metric(pred, true)

        self.log_dict(
            {
                f"test_{k}": v
                for k, v in results.items()
                if v is not None and not math.isnan(v)
            },
            on_epoch=True,
        )

        return results
