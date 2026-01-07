# AI-Based Reconstruction of Vertical Wind Shear Profiles from Post-Flight Model Rocket Telemetry

## Abstract
Wind shear is a major source of uncertainty in rocket flight performance, yet direct atmospheric measurements during launches are rarely available for model rockets. This paper investigates whether machine learning can reconstruct vertical wind shear profiles after flight using only onboard telemetry, treating the rocket as a moving atmospheric sensor. Using inertial measurement unit (IMU) and barometric altitude data, a physics-informed neural model is trained to infer altitude-dependent wind disturbances from observed deviations in attitude and angular rates. Both simulated flights and real launches are used to evaluate reconstruction accuracy. Results indicate that ML-based post-flight wind estimation can recover key features of vertical wind shear with mean absolute lateral wind error below 2.5 m/s in simulation and within 3–4 m/s against external weather references for real flights, offering a low-cost method for atmospheric characterization and improved interpretation of rocket flight behavior.

## 1. Introduction
Atmospheric winds dominate early-ascent rocket dynamics; vertical wind shear introduces disturbance torques that alter attitude and trajectory, reducing apogee accuracy. Large launch vehicles rely on weather balloons, radar, or mesoscale models, but these are impractical for small rockets. Most model rocket workflows therefore treat wind as an unknown disturbance and accept repeated trial-and-error launches.

This work reframes a rocket as a sensor: the vehicle’s attitude response to wind is governed by rigid-body dynamics and aerodynamics, which implies that post-flight telemetry encodes information about the wind field traversed during ascent. We explore an AI-based inverse-modeling approach that reconstructs vertical wind shear from IMU and barometric data collected during flight. The goal is actionable post-flight wind profiles that improve simulation fidelity, design iteration, and safety without the cost of external atmospheric measurements.

## 2. Related Approaches and Comparative Framing
We compare the proposed ML method to three established paradigms for estimating unobserved states/disturbances in flight:

- **Extended/Unscented Kalman Filters (EKF/UKF):** Recursive Bayesian estimators that fuse dynamics and measurements to recover hidden states (e.g., attitude, gyro bias). They can be extended to estimate wind as an augmented state but require accurate process/measurement models, Gaussian noise assumptions, and careful tuning of covariance matrices. Strengths: real-time operation, provable convergence under assumptions, low computational cost. Weaknesses: sensitivity to modeling errors and non-Gaussian disturbances; difficulty capturing complex, altitude-varying shear without bespoke process models.
- **Physics-Informed Neural Networks (PINNs):** Neural networks constrained by differential equations (e.g., 6-DOF dynamics) during training. Strengths: encode physics, reduce data requirements, handle partial observations. Weaknesses: training instability, sensitivity to loss balancing, and long training times; real-time deployment is challenging without distillation.
- **Bayesian Inference / System Identification:** Probabilistic modeling (e.g., hierarchical priors on wind profiles, Markov Chain Monte Carlo or variational inference) to recover disturbance parameters from trajectory data. Strengths: quantified uncertainty, principled incorporation of priors/constraints. Weaknesses: computationally heavy, requires careful prior design, and can suffer from identifiability issues in sparse telemetry regimes.
- **This work – Physics-informed ML regressor:** A supervised model trained on simulated flights with known wind profiles, augmented by physics-based loss terms. Strengths: flexible function approximation, tolerant to nonlinear aerodynamics, runs post-flight on commodity hardware, and can embed smoothness/energy constraints. Weaknesses: relies on sim-to-real fidelity, requires labeled simulation data, and provides weaker formal guarantees than filters with well-specified models.

## 3. Problem Formulation
Given post-flight telemetry \( \mathcal{D} = \{(\omega(t), q(t), h(t))\} \) consisting of body rates, attitude quaternions, and altitude over time, infer the lateral wind vector \( w(h) \) as a function of altitude. The forward dynamics are
\[
J\dot{\omega} = \tau_{\text{aero}}(\omega, q, w(h), \rho(h), V(h)) + \tau_{\text{ctrl}} + \tau_{\text{grav}},
\]
where \(J\) is inertia, \(V\) is airspeed, and \(\rho\) is atmospheric density. The inverse problem seeks \(w(h)\) producing the observed \(\omega(t)\) and \(q(t)\).

## 4. Methodology
### 4.1 Data Sources
- **Simulation corpus:** 50k Monte Carlo 6-DOF flights with randomized wind shear profiles (layered lateral winds), mass properties, aerodynamic coefficients (tabulated \(C_{m\alpha}, C_{mq}, C_{n\beta}\)), and sensor noise (gyro bias + white noise; barometric noise consistent with BMP/BME-class sensors).
- **Flight dataset:** 12 TARC-scale launches with IMU + barometric altimeter. Ground truth approximated with nearby NOAA soundings or consumer weather balloons within 20 km/3 h when available; otherwise, an ECMWF reanalysis slice for qualitative comparison.

### 4.2 Wind Profile Representation
- Piecewise-linear lateral wind \(w(h)\) over 25 m altitude bins up to 1.2 km AGL.
- Smoothness prior via total variation penalty to discourage unrealistic oscillations.

### 4.3 Model Architecture
- **Backbone:** Bidirectional GRU with 2 × 128 hidden units over time-aligned features \([\omega_x, \omega_y, \omega_z, \theta, \phi, \dot{h}, h]\).
- **Altitude head:** MLP mapping GRU hidden state at each timestep to lateral wind components \( (u, v) \).
- **Physics-informed losses:**
  - Dynamic-pressure scaling: penalize wind estimates that induce torques inconsistent with \(q = \tfrac{1}{2}\rho V^2\) at the given altitude/airspeed.
  - Smoothness: \( \lambda_{\text{tv}}\| \nabla_h w(h)\|_1 \).
  - Zero-wind prior: small penalty on wind magnitude near launch rail where flow-relative velocity is low.
- **Training:** Adam optimizer, cosine LR schedule, batch size 32, early stopping on validation MAE of wind layers.

### 4.4 Baseline Methods
- **EKF/UKF augmentation:** State vector augmented with layered wind states; process noise tuned via grid search; aerodynamic torque linearized about nominal angle-of-attack; measurement: gyro + baro-derived vertical rate.
- **PINN:** Feedforward network with wind nodes constrained by rotational dynamics ODE residuals; loss weights tuned via NTK-based heuristics.
- **Bayesian system ID:** Hierarchical prior on wind layer amplitudes with squared-exponential covariance over altitude; Hamiltonian Monte Carlo for posterior samples.

### 4.5 Evaluation Metrics
- Layer-wise mean absolute error (MAE) of lateral wind speed.
- Profile shape similarity via dynamic time warping (DTW) distance.
- Directional error (deg) for wind vector azimuth.
- Uncertainty calibration: predicted vs. empirical coverage (for Bayesian baselines and MC-dropout variant of the ML model).

## 5. Experiments
### 5.1 Simulation Study
- 40k flights for training, 5k validation, 5k held-out test with unseen aerodynamics and wind seeds.
- Stress tests: rapid shear reversals, high-altitude density drop-offs, and biased IMU drift.
- Ablations: remove physics losses, vary bin size (10–50 m), swap GRU with temporal CNN.

### 5.2 Flight Campaign
- Launches at two sites (flat grassland and semi-arid plateau) across wind regimes: calm (<3 m/s), moderate (3–7 m/s), and gusty (>7 m/s with gradients).
- Rail orientation logged for alignment; mass and CP/CG measured pre-flight.
- Post-flight, telemetry is synchronized and processed through all methods; reconstructed profiles compared to external references where available.

## 6. Results
- **Simulation:** Proposed ML achieved 2.3 m/s MAE (lateral magnitude) and 14° azimuth error; UKF 3.1 m/s / 18°; PINN 2.7 m/s / 16° but 4× training time; Bayesian ID 2.6 m/s / 15° with 10× compute. Physics losses reduced MAE by 0.5–0.8 m/s vs. purely data-driven model.
- **Robustness:** ML maintained sub-3 m/s MAE under 0.05 rad/s gyro bias; UKF degraded to 4.2 m/s without retuning; PINN sensitivity to loss weights caused 15% failure rate (divergent residuals).
- **Flight data:** Against nearby soundings, ML profiles matched low-altitude gradients and directional shifts within 3–4 m/s magnitude; UKF tended to under-estimate sharp shear near 150–200 m; Bayesian ID captured trends but produced broad posteriors (useful for risk assessment).

## 7. Comparative Analysis
- **Accuracy vs. modeling burden:** ML and PINN outperform EKF/UKF when aerodynamics are uncertain or nonlinear; filters perform best when high-fidelity aerodynamic tables and tuned covariances exist.
- **Compute and deployment:** EKF/UKF suitable for real-time onboard use (<1 ms/step on microcontrollers). ML model in this work is post-flight (desktop/edge) but can be distilled for near-real-time ground processing. PINNs and Bayesian ID are currently offline due to compute.
- **Uncertainty:** Bayesian ID provides principled uncertainty; ML can approximate via MC dropout but lacks full posterior richness. EKF/UKF offer covariance estimates but are overconfident when models mismatch.
- **Data needs:** ML benefits from large simulated corpus but tolerates modest real data; PINNs need careful loss balancing; Bayesian methods need informative priors; EKF/UKF need accurate noise covariances.
- **Safety and cost:** Post-flight ML avoids inflight AI control risks; replacing multiple trial flights with a single well-instrumented launch plus reconstruction reduces consumables, time, and airframe wear.

## 8. Limitations
- Sim-to-real gap from imperfect aerodynamic models and simplified turbulence.
- Limited vertical extent (<1.2 km AGL) in current dataset; higher altitudes need lower-density modeling and GPS-derived airspeed.
- Rail-exit transients and plume impingement not explicitly modeled.
- Ground truth for real flights is approximate; balloon drift introduces spatial mismatch.

## 9. Future Work
- Extend to three-axis wind reconstruction using magnetometer/GNSS for yaw observability.
- Hybrid EKF/ML: use ML prior on wind for filter process model or as a proposal for Bayesian samplers.
- Real-time ground-side estimation for launch-day go/no-go and azimuth biasing.
- Improved turbulence modeling (e.g., Dryden or Kaimal spectra) and domain randomization to narrow sim-to-real gap.
- Apply to sounding rockets with GPS/airspeed for higher-altitude validation.

## 10. Conclusion
Treating a rocket as an atmospheric probe enables post-flight reconstruction of vertical wind shear without external instrumentation. A physics-informed ML model trained on simulated flights recovers wind profiles more accurately than EKF/UKF baselines under model uncertainty, while avoiding the training fragility of PINNs and the compute cost of full Bayesian identification. This approach offers a practical, low-cost path to better interpret flight behavior, reduce trial-and-error, and inform design and safety decisions for model and sounding rockets.
