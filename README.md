## SOURCE FILES (directory `src`)

* `cv.py` – provides:
  * `cross_validate` - evaluates model quality using cross validation with *n_splits* (default: 10) folds 
  * `cv_relevance` and `cv_novelty` - evaluates relevance/novelty for all indepedent variables using CV (using both methods of density calibration)
  * `print_cv_relevance` and `print_cv_novelty` - for printing CV relevance/novelty results for all independent variables
  * `plot_cv_relevance` and `plot_cv_novelty` - for ploting mean CV relevance/novelty results for all independent variables
* `edf.py` – provides:
  * `edf_normalize` - function used for data normalization with the **Empirical Distribution Function**
  * `col_denorm` -  helpful for reversing the normalization of data from a selected column
* `evaluation.py` – provides:
  * `mean_log_likelihood` - function for assessing model quality
  * `evaluate_fold` - fuction useful for cross-validation
  * `relevance` - measures relevance of given independent variable as mean log-likelihood when only moment-like features for given colum are available 
  * `novelty` - measures novelty of given independent variable as defference beetween mean log-likelihood when all moment-like features are available and mean log-likelihood when moment-like freatures for given column are exluded
  * `expected_value` - for predicting the target value as the expected value of the modeled distribution
  * `mse_evaluate` - for computing the **Mean Squared Error** between the true target values and the values obtained from expectation-based predictions
* `features.py` – provides:
  * `moment_like_features` - function for creating features used in regression
  * `target_features` - function for creating target features used in regression
  * `prepare_targets` - prepars all target features from 1 to given degree N
* `hcr.py` – provides:
  * `fit_lasso` - function which implements HCR using L1-regularized regression
  * `make_density`- uses regression-derived models to obtain a density function
  * `calibrate_density` - for calibrating raw density functions
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

## RELEVANCE AND NOVELTY ANALYSIS
In the files `HL-relevance-and-novelty.ipynb` and `CL-relevance-and-novelty.ipynb`, an analysis of the relevance and novelty of individual independent variables was conducted, both in the basic version and using 10-fold cross-validation.
