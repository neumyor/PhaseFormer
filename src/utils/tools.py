import numpy as np
import torch
import os
import torch.nn as nn

# import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from datetime import datetime
from distutils.util import strtobool
from transformers.modeling_utils import Conv1D
import pandas as pd

from .metrics import metric

# plt.switch_backend("agg")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def convert_conv1d_to_linear(layer_conv: Conv1D, token_dim, split=False):
    conv_weights = layer_conv.weight
    conv_biases = layer_conv.bias

    if split:
        linear_weights = torch.split(
            conv_weights, split_size_or_sections=token_dim, dim=1
        )
        linear_biases = torch.split(conv_biases, split_size_or_sections=token_dim)

        # replace the self attn
        layer_linear_list = [
            nn.Linear(token_dim, token_dim) for _ in range(len(linear_weights))
        ]
        for i in range(len(linear_weights)):
            layer_linear_list[i].weight.data = linear_weights[i].transpose(0, 1)
            layer_linear_list[i].bias.data = linear_biases[i]

        return layer_linear_list
    else:
        layer_linear = nn.Linear(token_dim, token_dim)
        layer_linear.weight.data = conv_weights.transpose(0, 1)
        layer_linear.bias.data = conv_biases
        return layer_linear


def convert_linear_list_to_conv1d(layer_linear, token_dim):
    conv_layer = Conv1D(token_dim * len(layer_linear), token_dim)
    if isinstance(layer_linear, list):
        weights = []
        bias = []
        for linear_layer in layer_linear:
            weights.append(linear_layer.weight.data.transpose(0, 1))
            bias.append(linear_layer.bias.data)
        weights = torch.concatenate(weights, dim=1)
        bias = torch.concatenate(bias)

        conv_layer.weight.data = weights
        conv_layer.bias.data = bias

        return conv_layer

    else:
        raise NotImplementedError("do it yourself")
        pass


def freeze_module(module, trainable_marks):
    for name, param in module.named_parameters():
        param.requires_grad = False
        for mark in trainable_marks:
            if mark in name:
                param.requires_grad = True
                break


def bridge_token_init(mode, shared, num_variants, num_bridge_tokens, dim):
    shape_prefix = (num_bridge_tokens,) if shared else (num_variants, num_bridge_tokens)

    if mode == "zero":
        tokens = torch.empty((*shape_prefix, dim))
        nn.init.uniform_(tokens, -0.02, 0.02)
    elif mode == "zeros":
        tokens = torch.empty((*shape_prefix, dim))
        torch.nn.init.normal_(tokens, mean=0.0, std=0.1)
    else:
        raise NotImplementedError(f"{mode} is not a valid mode for bridge_token_init")

    return nn.Parameter(tokens)


def adjust_learning_rate(optimizer, epoch, lr_config, lr):
    lradj = lr_config["type"]
    if lradj == "type1":
        lr_adjust = {epoch: lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1))}
    elif lradj == "type2":
        lr_adjust = {epoch: lr * (lr_config.decay_fac ** ((epoch - 1) // 1))}
    elif lradj == "type4":
        lr_adjust = {epoch: lr * (lr_config.decay_fac ** ((epoch) // 1))}
    else:
        lr_adjust = {epoch: lr}

    print("lr_adjust = {}".format(lr_adjust))

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.delta = delta
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is not None and score < self.best_score + self.delta:
            self.counter += 1
            is_best = False
            if self.counter >= self.patience:
                is_stop = True
            else:
                is_stop = False
        else:
            self.best_score = score
            self.counter = 0
            is_stop = False
            is_best = True
            self.val_loss_min = val_loss

        return {
            "is_stop": is_stop,
            "is_best": is_best,
            "vali_loss_min": self.val_loss_min,
            "patience": self.patience,
            "counter": self.counter,
        }

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf

    # def save_checkpoint(self, val_loss, model, path, epoch):
    #     if not self.accelerator.is_main_process:
    #         return
    #     if self.verbose:
    #         print(
    #             f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
    #         )

    #     if self.accelerator:
    #         unwrapped_model = self.accelerator.unwrap_model(model)
    #         self.accelerator.save(
    #             unwrapped_model.state_dict(), path + "/" + "checkpoint.pth"
    #         )
    #     else:
    #         torch.save(model.state_dict(), path + "/" + "checkpoint.pth")

    #     file_name = f"valid_loss_{str(round(val_loss.item(), 4))}_epoch-{epoch}"
    #     file_path = os.path.join(path, file_name)

    #     with open(file_path, "w") as file:
    #         pass

    #     self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    raise NotImplementedError("visual() is not implemented")
    # plt.figure()
    # plt.plot(true, label="GroundTruth", linewidth=2)
    # if preds is not None:
    #     plt.plot(preds, label="Prediction", linewidth=2)
    # plt.legend()
    # plt.savefig(name, bbox_inches="tight")


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))
