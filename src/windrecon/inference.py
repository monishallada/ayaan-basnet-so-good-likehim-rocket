import argparse
import os
import csv

import numpy as np
import torch

from windrecon.config import ModelConfig
from windrecon.model.gru_wind_estimator import GRUWindEstimator, _bin_predictions


def load_checkpoint(path: str) -> GRUWindEstimator:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = GRUWindEstimator(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def reconstruct(model: GRUWindEstimator, feature: np.ndarray, length: int, altitude_bins: np.ndarray, mc_samples: int = 1):
    device = next(model.parameters()).device
    feat = torch.from_numpy(feature[None, ...]).float().to(device)
    lengths = torch.tensor([length], device=device)
    bin_size = float(altitude_bins[1] - altitude_bins[0]) if len(altitude_bins) > 1 else 25.0

    preds = []
    for _ in range(mc_samples):
        if mc_samples > 1:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            wind_ts = model(feat)
            binned = _bin_predictions(
                wind_ts,
                feat[:, :, -1],
                lengths,
                bin_size,
                model.cfg.altitude_bins,
            )
            preds.append(binned.squeeze(0).cpu().numpy())
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0) if mc_samples > 1 else np.zeros_like(mean_pred)
    return mean_pred, std_pred


def main():
    parser = argparse.ArgumentParser(description="Infer vertical wind profile from telemetry.")
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--data", required=True, help=".npz dataset path")
    parser.add_argument("--index", type=int, default=0, help="Flight index within dataset")
    parser.add_argument("--mc", type=int, default=1, help="MC dropout samples for uncertainty")
    parser.add_argument("--out", required=True, help="CSV output path")
    args = parser.parse_args()

    data = np.load(args.data)
    features = data["features"]
    winds = data["winds"]
    lengths = data["lengths"]
    altitude_bins = data["altitude_bins"]

    model = load_checkpoint(args.model)

    feature = features[args.index]
    length = int(lengths[args.index])
    pred, std = reconstruct(model, feature, length, altitude_bins, mc_samples=args.mc)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["altitude_m", "wind_u_mps", "wind_v_mps", "std_u", "std_v", "true_u", "true_v"])
        for i, alt in enumerate(altitude_bins):
            true_u, true_v = winds[args.index, i]
            writer.writerow([alt, pred[i, 0], pred[i, 1], std[i, 0], std[i, 1], true_u, true_v])
    print(f"Saved wind reconstruction to {args.out}")


if __name__ == "__main__":
    main()
