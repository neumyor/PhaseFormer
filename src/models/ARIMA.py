import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


class ARIMAModel(nn.Module):
    def __init__(self, p=5, d=1, q=0, pred_len=10):
        super(ARIMAModel, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.pred_len = pred_len

    def forward(self, x_enc):
        x = x_enc
        bsz, seq_len, num_channels = x.shape
        preds = torch.zeros(bsz, self.pred_len, num_channels)
        
        for i in range(bsz):
            for j in range(num_channels):
                series = x[i, :, j].detach().cpu().numpy()
                model = ARIMA(series, order=(self.p, self.d, self.q))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=self.pred_len)
                preds[i, :, j] = torch.tensor(forecast)
        
        return preds