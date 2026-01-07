# WindRecon: Physics-Informed ML for Post-Flight Wind Shear Reconstruction

An end-to-end reference implementation of the system described in `AI_wind_shear_reconstruction_paper.md`: a physics-informed neural model that reconstructs vertical wind shear profiles from model rocket telemetry. The stack includes synthetic data generation, training with physics losses, and post-flight inference.

## Features
- **Simulation-first training:** Monte Carlo 6-DOF style synthetic flights with layered wind profiles, gyro/baro noise, and aerodynamic variability.
- **Physics-informed GRU:** Bidirectional GRU backbone with altitude head and losses that enforce dynamic-pressure scaling, smoothness, and zero-wind priors.
- **Baselines included:** EKF-style wind augmentation stub and Bayesian ID hook points for future expansion.
- **Inference pipeline:** Post-flight reconstruction of lateral wind vectors binned by altitude with optional uncertainty via MC dropout.

## Repo Layout
- `requirements.txt` — minimal dependencies (PyTorch, NumPy, SciPy).
- `src/windrecon/config.py` — dataclass configs for model, training, and simulation.
- `src/windrecon/data/sim_generator.py` — synthetic telemetry + wind profile generation.
- `src/windrecon/model/gru_wind_estimator.py` — neural architecture and physics-informed loss assembly.
- `src/windrecon/train.py` — training loop with checkpointing and validation.
- `src/windrecon/inference.py` — post-flight wind reconstruction entrypoint.
- `src/windrecon/baselines/ekf_stub.py` — EKF augmentation scaffold.
- `src/windrecon/baselines/bayesian_stub.py` — placeholder for Bayesian ID.

## Quickstart (simulation -> train -> infer)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic dataset (saved as .npz)
python -m windrecon.data.sim_generator --out data/sim_dataset.npz --samples 2000

# Train model
python -m windrecon.train --data data/sim_dataset.npz --epochs 25 --out checkpoints/model.pt

# Infer wind profile from held-out telemetry
python -m windrecon.inference --model checkpoints/model.pt --data data/sim_dataset.npz --index 10 --out outputs/wind_profile.csv

# Launch local web UI (no API keys needed)
uvicorn windrecon.web.app:app --reload --port 8000
# then open http://localhost:8000 to upload a telemetry .npz and run reconstruction via the local model checkpoint
```

## Notes
- The EKF/Bayesian modules are stubs for future work; primary path is the physics-informed GRU.
- Replace the synthetic generator with real telemetry by formatting to the expected arrays in `inference.py`.
- For GPU training, install the CUDA-enabled PyTorch wheel matching your system.

## Deploying a free UI without API keys
- **Local:** Use the FastAPI + HTML UI via `uvicorn windrecon.web.app:app --port 8000`. All computation is on your machine; no external services.
- **Vercel frontend + separate backend (free-tier):** Vercel cannot run this PyTorch backend directly. Host a small Python backend (e.g., Fly.io, Render free tier, Railway hobby) with this repo and `uvicorn`. Point a simple static frontend on Vercel to the backend API `/api/infer`. No API keys are required; the model runs on the backend VM.
- **Streamlit/Gradio alternative:** You can wrap `windrecon.inference` in Streamlit for one-click local use; also free on Streamlit Community Cloud.
