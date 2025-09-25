import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from src.utils.metrics import metric


class DefaultPLModule(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.args = configs

        self.target_var_index = int(configs.get("target_var_index", -1))
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt_cls = torch.optim.Adam
        optimizer = opt_cls(self.parameters(), lr=self.args.training_args.learning_rate)
        if self.args.training_args.lr_schedule_config.type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.training_args.lr_schedule_config.tmax,
                eta_min=1e-8,
            )
            return [optimizer], [scheduler]
        return optimizer

    def _build_decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(
            batch_y[:, -self.args.dataset_args.pred_len :, :]
        ).float()
        dec_inp = torch.cat(
            [batch_y[:, : self.args.dataset_args.label_len, :], dec_inp], dim=1
        ).float()
        return dec_inp

    def _get_criterion(self, loss_type):
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "mae":
            criterion = nn.L1Loss()
        elif loss_type == "smae":
            criterion = nn.SmoothL1Loss()
        elif loss_type == "smape":

            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()

                def forward(self, pred, true):
                    return torch.mean(
                        200
                        * torch.abs(pred - true)
                        / (torch.abs(pred) + torch.abs(true) + 1e-8)
                    )

            criterion = SMAPE()
        else:
            raise ValueError(
                f"loss function {self.args.training_args.loss_func} not supported yet"
            )
        return criterion

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

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)

        loss = criterion(outputs, batch_y)
        self.log("train_loss", loss, on_epoch=True)

        return loss

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
        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
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

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        pred = outputs.detach()
        true = batch_y.detach()

        loss = metric(pred, true)
        self.log_dict({f"test_{k}": v for k, v in loss.items()}, on_epoch=True)

        return loss
