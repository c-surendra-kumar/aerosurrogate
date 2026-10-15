| Model | Cl MAE | Cl rel. error (%) | Cl R2 | Cd MAE | Cd rel. error (%) | Cd R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean predictor | 0.4975 | 147.85 | -0.0001 | 0.003352 | 25.18 | -0.0106 |
| Linear baseline | 0.0205 | 5.18 | 0.9976 | 0.000793 | 5.85 | 0.9194 |
| **MLP (this work)** | 0.0090 | 2.30 | 0.9995 | 0.000303 | 2.16 | 0.9838 |

Paired bootstrap over the 200 test simulations (10,000 resamples):

- **Cl**: the MLP's MAE is lower by 0.011403, 95% CI [0.008820, 0.014099], winning 100.0% of resamples (significant).
- **Cd**: the MLP's MAE is lower by 0.000490, 95% CI [0.000381, 0.000610], winning 100.0% of resamples (significant).
