# ML Surrogate Modeling for Airfoil Aerodynamics

Supervised ML surrogate models that map **CST airfoil coefficients + angle of attack** to aerodynamic coefficients (C_L, C_D, C_M), replacing high-cost CFD runs with millisecond-scale inference.

## Models Benchmarked

| Model             | Library      |
| ----------------- | ------------ |
| Linear Regression | Scikit-learn |
| Decision Tree     | Scikit-learn |
| Random Forest     | Scikit-learn |
| Neural Network    | PyTorch      |

Train/test split is on **held-out airfoil geometries** — no data leakage across shapes.


## Data

- `long_polar_with_CST.xlsx` - 129k samples, 18 CST coeffs, Re, α → C_L, C_D, C_M
- Outputs
- `airfoil_surrogate_nn.pt` - PyTorch checkpoint (with scalers)
- `airfoil_surrogate_rf.joblib` - Random Forest model
