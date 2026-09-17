# Logistic Regression of BOVA11 Returns via Newton-Raphson Optimization

A numerical optimization approach to binary classification of daily log-returns for **BOVA11** (an ETF tracking the Brazilian iBovespa index). 

Built **entirely from scratch** using numerical linear algebra in Python, this project models market direction using an Iteratively Reweighted Least Squares (IRLS) scheme powered by second-order **Newton-Raphson** optimization. Instead of using high-level machine learning frameworks like `scikit-learn`, all gradient and Hessian computations are vectorized directly with `numpy`.

---

## Model Formulation

Standard linear models struggle with non-linear probability mapping in binary classification. We model the conditional probability $p_t = P(y_t = 1 \mid X_t)$ that the next-day log-return is positive ($y_t = 1$) given a vector of lagged daily returns $X_t$:

$$p_t = \sigma(X_t \beta) = \frac{1}{1 + e^{-X_t \beta}}$$

Where $X_t = [1, x_{t-1}, x_{t-2}, \dots, x_{t-k}]$ contains the intercept and $k$ autoregressive lag features.

### Optimization Scheme (IRLS / Newton-Raphson)
To minimize the negative log-likelihood function, we update the parameter vector $\beta \in \mathbb{R}^{k+1}$ iteratively using second-order Taylor expansion:

$$\beta^{(m+1)} = \beta^{(m)} - H^{-1} g$$

* **Gradient Vector ($g$):** $g = X^T (p - y) \in \mathbb{R}^{k+1}$ — First derivative measuring log-likelihood sensitivity.
* **Weight Matrix ($W$):** $W = \text{diag}(p_i (1 - p_i)) \in \mathbb{R}^{N \times N}$ — Variance weight matrix for Bernoulli trials.
* **Hessian Matrix ($H$):** $H = X^T W X \in \mathbb{R}^{(k+1) \times (k+1)}$ — Second derivative matrix capturing local curvature.

The implementation vectorizes $H$ as $X^T (X \odot w)$, where $w_i = p_i(1-p_i)$ and $\odot$ represents row-wise broadcasting, avoiding explicit allocation of large diagonal matrices.

---

## Key Findings

Fitting the model on historical BOVA11 daily return lags ($k=3$) yields quadratic convergence in **4 iterations**:

* **$\beta_0$ (Intercept):** Estimated at $0.0520$. Represents a slight baseline positive drift in daily return direction.
* **$\beta_1$ (Lag 1):** Concentrated around $8.6034$. Demonstrates a strong short-term inertia/momentum effect, where yesterday's positive return increases the probability of a positive return today.
* **$\beta_2$ (Lag 2):** Estimated at $1.5094$. Indicates a moderate continuation effect from two days prior.
* **$\beta_3$ (Lag 3):** Concentrated around $-6.2364$. Captures a strong mean-reversion behavior, counteracting multi-day directional trends.
* **Model Accuracy:** The training accuracy settled around $53.42\%$, reflecting the high signal-to-noise ratio and partial weak-form efficiency typical of financial equity indices.

---

## Dependencies

* Python 3.x
* `numpy`
* `pandas`
* `yfinance`

Install required packages:
```bash
pip install numpy pandas yfinance
