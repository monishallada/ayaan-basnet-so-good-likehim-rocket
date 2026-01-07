import argparse
import math
import os
from typing import Tuple

import numpy as np

from windrecon.config import SimulationConfig


def _air_density(alt_m: np.ndarray) -> np.ndarray:
    sea_level_density = 1.225
    scale_height = 8500.0
    return sea_level_density * np.exp(-alt_m / scale_height)


def _generate_wind_profile(cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    bins = int(cfg.max_altitude_m // cfg.altitude_bin_m)
    alt_centers = np.arange(bins) * cfg.altitude_bin_m + cfg.altitude_bin_m * 0.5
    base = np.linspace(0, 1, bins)
    gust = np.sin(np.linspace(0, math.pi * 1.5, bins))
    wind_mag = cfg.wind_mean_mps + cfg.wind_std_mps * (0.3 * gust + 0.7 * np.random.randn(bins))
    wind_mag = np.clip(wind_mag, 0, cfg.wind_mean_mps + 3 * cfg.wind_std_mps)
    direction = np.random.uniform(0, 2 * math.pi)
    wind_u = wind_mag * np.cos(direction)
    wind_v = wind_mag * np.sin(direction)
    profile = np.stack([wind_u, wind_v], axis=-1)
    smooth_profile = np.convolve(profile[:, 0], [0.2, 0.6, 0.2], mode="same")
    profile[:, 0] = smooth_profile
    return alt_centers, profile


def _simulate_flight(cfg: SimulationConfig, alt_centers: np.ndarray, wind_profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dt = cfg.dt
    steps = int(cfg.max_altitude_m / 40.0 / dt)  # simple ascent profile ~40 m/s average
    h = np.zeros(steps)
    dh_dt = np.zeros(steps)
    omega = np.zeros((steps, 3))
    theta_phi = np.zeros((steps, 2))

    inertia = np.diag([0.12, 0.12, 0.02])
    aero_gain = 0.5
    pitch_damper = 0.85

    thrust_time = int(2.0 / dt)
    thrust_accel = 60.0
    drag_coeff = 0.15

    for i in range(1, steps):
        accel = thrust_accel if i < thrust_time else -drag_coeff * dh_dt[i - 1]
        dh_dt[i] = max(0.0, dh_dt[i - 1] + accel * dt)
        h[i] = h[i - 1] + dh_dt[i] * dt
        if h[i] >= cfg.max_altitude_m:
            h = h[: i + 1]
            dh_dt = dh_dt[: i + 1]
            omega = omega[: i + 1]
            theta_phi = theta_phi[: i + 1]
            break

        wind = np.interp(h[i], alt_centers, wind_profile[:, 0]), np.interp(h[i], alt_centers, wind_profile[:, 1])
        wind_vec = np.array(wind)
        dyn_pressure = 0.5 * _air_density(np.array([h[i]]))[0] * max(dh_dt[i], 1.0) ** 2
        torque = aero_gain * dyn_pressure * wind_vec
        alpha = (np.linalg.inv(inertia) @ np.array([torque[0], torque[1], 0.0]).T).flatten()
        omega[i] = omega[i - 1] + alpha * dt
        omega[i] += np.random.randn(3) * 0.1
        theta_phi[i] = theta_phi[i - 1] + omega[i, :2] * dt - pitch_damper * theta_phi[i - 1] * dt

    gyro_bias = np.random.randn(3) * cfg.gyro_bias_std
    gyro_noise = np.random.randn(*omega.shape) * cfg.gyro_noise_std
    baro_noise = np.random.randn(h.shape[0]) * cfg.baro_noise_std

    omega_meas = omega + gyro_bias + gyro_noise
    h_meas = h + baro_noise

    features = np.column_stack(
        [omega_meas[:, 0], omega_meas[:, 1], omega_meas[:, 2], theta_phi[:, 0], theta_phi[:, 1], dh_dt, h_meas]
    )
    return features, wind_profile


def generate_dataset(cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alt_centers = None
    feature_list = []
    wind_list = []
    for _ in range(cfg.samples):
        alt_centers, wind_profile = _generate_wind_profile(cfg)
        features, profile = _simulate_flight(cfg, alt_centers, wind_profile)
        feature_list.append(features)
        wind_list.append(profile)
    # Pad sequences to max length for batching; shorter sequences are padded with zeros.
    max_len = max(f.shape[0] for f in feature_list)
    padded = np.zeros((cfg.samples, max_len, feature_list[0].shape[1]))
    lengths = np.zeros(cfg.samples, dtype=np.int32)
    for i, f in enumerate(feature_list):
        padded[i, : f.shape[0]] = f
        lengths[i] = f.shape[0]
    return padded, np.stack(wind_list, axis=0), alt_centers, lengths


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic rocket telemetry and wind profiles.")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--samples", type=int, default=1000, help="Number of flights")
    args = parser.parse_args()
    cfg = SimulationConfig(samples=args.samples)
    features, winds, alt_centers, lengths = generate_dataset(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, features=features, winds=winds, lengths=lengths, altitude_bins=alt_centers, dt=cfg.dt)
    print(f"Saved dataset to {args.out} with shape features={features.shape}, winds={winds.shape}, lengths={lengths.shape}")


if __name__ == "__main__":
    main()
