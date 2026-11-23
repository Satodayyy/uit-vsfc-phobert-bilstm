"""Baseline UIT-VSFC: đặc trưng TF-IDF và hồi quy logistic đa lớp bằng PyTorch."""
import time
from uuid import uuid4

import joblib
import numpy as np
import torch
from filelock import FileLock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F

from .chi_so import diem
from .du_lieu import IDS, OUT, SEEDS, checksum, ghi, luc_nay

LUOI_C = (0.1, 1.0, 10.0)
MAX_ITER = 250


def tensor_thua(matrix):
    coo = matrix.tocoo(copy=False)
    indices = torch.from_numpy(np.stack((coo.row, coo.col)).astype(np.int64, copy=False))
    values = torch.as_tensor(coo.data, dtype=torch.float64)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


class HoiQuyLogistic(nn.Module):
    def __init__(self, so_dac_trung, cac_nhan):
        super().__init__()
        self.classes_ = np.asarray(cac_nhan, dtype=np.int64)
        self.linear = nn.Linear(so_dac_trung, len(self.classes_), dtype=torch.float64)

    def forward(self, x):
        return torch.sparse.mm(x, self.linear.weight.T) + self.linear.bias

    @torch.inference_mode()
    def predict(self, x):
        x = x if isinstance(x, torch.Tensor) else tensor_thua(x)
        return self.classes_[self(x).argmax(dim=1).cpu().numpy()]

    @property
    def coef_(self):
        return self.linear.weight.detach().cpu().numpy()

    @property
    def intercept_(self):
        return self.linear.bias.detach().cpu().numpy()

def huan_luyen(x, y, cac_nhan, trong_so_lop, c, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = HoiQuyLogistic(x.shape[1], cac_nhan)
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=MAX_ITER, history_size=20,
        tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn="strong_wolfe",
    )
    # Cross-entropy lấy trung bình có trọng số. Tổng trọng số mẫu bằng số câu train.
    # C càng nhỏ thì phạt L2 càng mạnh. Chỉ phạt trọng số, giữ tự do cho hệ số chặn.
    def muc_tieu():
        entropy = F.cross_entropy(model(x), y, weight=trong_so_lop)
        return entropy + model.linear.weight.square().sum() / (2 * c * len(y))

    def closure():
        optimizer.zero_grad()
        loss = muc_tieu()
        loss.backward()
        return loss

    loss_dau = optimizer.step(closure).item()
    with torch.no_grad():
        loss_cuoi = muc_tieu().item()
    trang_thai = optimizer.state[model.linear.weight]
    return model.eval(), {
        "loss_dau": loss_dau, "loss_cuoi": loss_cuoi,
        "so_buoc_lbfgs": trang_thai["n_iter"], "so_lan_tinh_loss_lbfgs": trang_thai["func_evals"],
    }