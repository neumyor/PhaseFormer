import torch
import torch.nn as nn
import torch.nn.functional as F


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return F.mse_loss(pred, true)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return torch.mean(torch.abs(100 * (pred - true) / (true + 1e-8)))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / (true + 1e-8)))


def SMAPE(pred, true):
    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    # return torch.mean(200 * torch.abs(pred - true) / (pred + true + 1e-8))


def ND(pred, true):
    return torch.mean(torch.abs(true - pred)) / torch.mean(torch.abs(true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }