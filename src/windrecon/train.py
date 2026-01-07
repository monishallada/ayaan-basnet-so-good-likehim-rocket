import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from windrecon.config import ModelConfig, TrainingConfig
from windrecon.model.gru_wind_estimator import GRUWindEstimator, physics_informed_loss


class WindDataset(Dataset):
    def __init__(self, features: np.ndarray, winds: np.ndarray, lengths: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.winds = torch.from_numpy(winds).float()
        self.lengths = torch.from_numpy(lengths).long()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.winds[idx], self.lengths[idx]


def train_epoch(model, loader, optimizer, device, cfg: ModelConfig, bin_size: float):
    model.train()
    total_loss = 0.0
    for features, winds, lengths in loader:
        features, winds, lengths = features.to(device), winds.to(device), lengths.to(device)
        optimizer.zero_grad()
        pred = model(features)
        altitudes = features[:, :, -1]
        vertical_rate = features[:, :, -2]
        loss, _ = physics_informed_loss(cfg, pred, altitudes, vertical_rate, lengths, winds, bin_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device, cfg: ModelConfig, bin_size: float):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, winds, lengths in loader:
            features, winds, lengths = features.to(device), winds.to(device), lengths.to(device)
            pred = model(features)
            altitudes = features[:, :, -1]
            vertical_rate = features[:, :, -2]
            loss, _ = physics_informed_loss(cfg, pred, altitudes, vertical_rate, lengths, winds, bin_size)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train physics-informed wind reconstruction model.")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True, help="Checkpoint output path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = np.load(args.data)
    features = data["features"]
    winds = data["winds"]
    lengths = data["lengths"]
    altitude_bins = data["altitude_bins"]
    bin_size = float(altitude_bins[1] - altitude_bins[0]) if len(altitude_bins) > 1 else 25.0

    ds = WindDataset(features, winds, lengths)
    val_size = max(1, int(0.1 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    mcfg = ModelConfig(altitude_bins=winds.shape[1])
    tcfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_path=args.out,
    )

    device = torch.device(tcfg.device)
    model = GRUWindEstimator(mcfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.learning_rate)

    best_loss = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(tcfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, mcfg, bin_size)
        val_loss = eval_epoch(model, val_loader, device, mcfg, bin_size)
        print(f"Epoch {epoch+1}/{tcfg.epochs}: train {train_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "config": mcfg.__dict__}, args.out)
            print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
