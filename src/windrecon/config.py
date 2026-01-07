from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    samples: int = 1000
    max_altitude_m: float = 1200.0
    altitude_bin_m: float = 25.0
    wind_mean_mps: float = 4.0
    wind_std_mps: float = 2.0
    gyro_bias_std: float = 0.05  # rad/s
    gyro_noise_std: float = 0.02  # rad/s
    baro_noise_std: float = 1.5  # meters
    dt: float = 0.02  # 50 Hz telemetry


@dataclass
class ModelConfig:
    input_size: int = 7  # omega x/y/z, theta, phi, dh/dt, h
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    altitude_bins: int = 48  # 25 m bins to ~1200 m
    tv_weight: float = 0.01
    dyn_pressure_weight: float = 0.05
    zero_wind_weight: float = 0.002


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cpu"
    checkpoint_path: Optional[str] = None
