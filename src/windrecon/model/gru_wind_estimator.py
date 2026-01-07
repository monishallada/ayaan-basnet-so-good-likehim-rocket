from typing import Tuple

import torch
import torch.nn as nn

from windrecon.config import ModelConfig


class GRUWindEstimator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        wind_pred = self.head(out)
        return wind_pred


def _bin_predictions(wind_pred: torch.Tensor, altitudes: torch.Tensor, lengths: torch.Tensor, bin_size: float, num_bins: int):
    device = wind_pred.device
    bsz, max_len, _ = wind_pred.shape
    bins = torch.zeros((bsz, num_bins, 2), device=device)
    counts = torch.zeros((bsz, num_bins, 1), device=device)

    bin_idx = torch.clamp((altitudes / bin_size).long(), 0, num_bins - 1)
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

    for b in range(bsz):
        idx = bin_idx[b, mask[b]]
        wp = wind_pred[b, mask[b]]
        bins[b].index_add_(0, idx, wp)
        counts[b].index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32).unsqueeze(-1))

    bins = torch.where(counts > 0, bins / counts.clamp(min=1.0), bins)
    return bins


def physics_informed_loss(
    cfg: ModelConfig,
    wind_pred: torch.Tensor,
    altitudes: torch.Tensor,
    vertical_rate: torch.Tensor,
    lengths: torch.Tensor,
    target_bins: torch.Tensor,
    bin_size: float,
) -> Tuple[torch.Tensor, dict]:
    num_bins = cfg.altitude_bins
    binned_pred = _bin_predictions(wind_pred, altitudes, lengths, bin_size, num_bins)
    l1 = (binned_pred - target_bins).abs().mean()

    tv = (binned_pred[:, 1:] - binned_pred[:, :-1]).abs().mean()

    rho0 = 1.225
    scale_height = 8500.0
    rho = rho0 * torch.exp(-altitudes / scale_height)
    dyn_q = 0.5 * rho * torch.clamp(vertical_rate, min=1.0) ** 2
    q_norm = torch.tanh(dyn_q / 50.0)
    dyn_penalty = (wind_pred.norm(dim=-1) * (1 - q_norm)).mean()

    near_ground_bins = min(3, num_bins)
    zero_prior = wind_pred[:, :near_ground_bins].norm(dim=-1).mean() if wind_pred.size(1) >= near_ground_bins else torch.tensor(0.0, device=wind_pred.device)

    loss = (
        l1
        + cfg.tv_weight * tv
        + cfg.dyn_pressure_weight * dyn_penalty
        + cfg.zero_wind_weight * zero_prior
    )
    terms = {
        "l1": l1.detach(),
        "tv": tv.detach(),
        "dyn": dyn_penalty.detach(),
        "zero": zero_prior.detach(),
    }
    return loss, terms
