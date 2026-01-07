## SOURCE FILES (directory `src`)

* `edf.py` – provides the function `edf_normalize` used for data normalization with the **Empirical Distribution Function**, and `col_denorm`, which is helpful for reversing the normalization of data from a selected column
* `evaluation.py` – provides the function `mean_log_likelihood` for assessing model quality and `evaluate_fold` for use during cross-validation; additionally, it provides the function `expected_value` for predicting the target value as the expected value of the modeled distribution, and `mse_evaluate` for computing the **Mean Squared Error** between the true target values and the values obtained from expectation-based predictions
* `features.py` – provides the functions `moment_like_features` and `target_features` for creating features used in regression
* `hcr.py` – provides the function `fit_lasso`, which implements HCR using L1-regularized regression, `make_density`, which uses regression-derived models to obtain a density function, and `calibrate_density` for calibrating raw density functions
* `legendre.py` – provides the class `RescaledLegendre`, which implements an orthonormal basis of rescaled Legendre polynomials

## BASIC USAGE

The files `HL-basic.ipynb` and `CL-basic.ipynb` contain an implementation of the basic version of HCR using the source files described above to model probability density functions for:

* `Heating Load` – energy demand for heating
* `Cooling Load` – energy demand for cooling <br>

based on independent variables from the **Energy Efficiency** dataset. In this version, the data are split into training and test sets in a single way, with proportions **90%** for the training set and **10%** for the test set.

The obtained *mean log-likelihood* on the test set is:

* HL (calibration: *softplus*): **0.7688**
* HL (calibration: *clip*): **1.0098**
* CL (calibration: *softplus*): **0.7556**
* CL (calibration: *clip*): **0.9897**

Additionally, the notebooks include an example of predicting the target variable as the expected value of the modeled distribution and reversing the normalization. At the end, evaluation using MSE on the test set is presented.

## 10-fold CV

In the file `CV.ipynb`, the operations from `HL-basic.ipynb` and `CL-basic.ipynb` are repeated, but using cross-validation with a split into 10 folds.

The obtained results for `Heating Load`:

* calibration method: **softplus** <br>
  per fold: [0.71 0.69 0.74 0.71 0.76 0.76 0.66 0.7  0.71 0.65] <br>
  mean LL : `0.7100` <br>
  std LL  : 0.0354

* calibration method: **clip** <br>
  per fold: [0.94 0.89 0.98 0.92 0.83 0.99 0.88 0.93 0.93 0.71] <br>
  mean LL : `0.8985` <br>
  std LL  : 0.0770

The obtained results for `Cooling Load`:

* calibration method: **softplus** <br>
  per fold: [0.73 0.73 0.65 0.74 0.75 0.69 0.66 0.69 0.71 0.64] <br>
  mean LL : `0.6993` <br>
  std LL  : 0.0385

* calibration method: **clip** <br>
  per fold: [0.97 0.96 0.88 0.98 0.81 0.92 0.88 0.91 0.93 0.84] <br>
  mean LL : `0.9074` <br>
  std LL  : 0.0537
