# 📐 Linear Regression — Complete Notes with Math Intuition

> From first principles to gradient descent, with every derivation shown step by step.

---

## Table of Contents

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [The Hypothesis Function](#2-the-hypothesis-function)
3. [Understanding θ₀ and θ₁ (Theta 0 & Theta 1)](#3-understanding-θ₀-and-θ₁)
4. [The Cost Function J(θ)](#4-the-cost-function-jθ)
5. [Math Intuition — Minimizing Cost](#5-math-intuition--minimizing-cost)
6. [Gradient Descent](#6-gradient-descent)
7. [Deriving the Gradient Descent Update Rules](#7-deriving-the-gradient-descent-update-rules)
8. [The Normal Equation (Closed-Form Solution)](#8-the-normal-equation-closed-form-solution)
9. [Full Algorithm Summary](#9-full-algorithm-summary)
10. [Python Implementation from Scratch](#10-python-implementation-from-scratch)
11. [Key Hyperparameters & Pitfalls](#11-key-hyperparameters--pitfalls)

---

## 1. What is Linear Regression?

Linear Regression models the **linear relationship** between one or more input features **x** and a continuous output **y**.

Given a dataset of `m` training examples:

```
{ (x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), ..., (x⁽ᵐ⁾, y⁽ᵐ⁾) }
```

Our goal: Find a line (or hyperplane) that **best fits** the data — i.e., minimizes prediction error.

### Simple vs. Multiple Linear Regression

| Type | Features | Equation |
|---|---|---|
| Simple | 1 feature (x) | ŷ = θ₀ + θ₁x |
| Multiple | n features (x₁…xₙ) | ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ |

---

## 2. The Hypothesis Function

For **Simple Linear Regression**, the hypothesis is:

```
h_θ(x) = θ₀ + θ₁ · x
```

Where:
- `h_θ(x)` = predicted output (ŷ)
- `θ₀` = **bias / intercept** — shifts the line up or down
- `θ₁` = **weight / slope** — controls the steepness of the line
- `x` = input feature

For **Multiple Linear Regression** with `n` features:

```
h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

In **vectorized form** (adding x₀ = 1 as a bias term):

```
h_θ(x) = θᵀ · x

where θ = [θ₀, θ₁, ..., θₙ]ᵀ  and  x = [1, x₁, x₂, ..., xₙ]ᵀ
```

---

## 3. Understanding θ₀ and θ₁

### 🎯 Geometric Intuition

```
y
│                          ● (actual point)
│                    ●    ↑ error (residual)
│             ●    ──────╴predicted point on line
│        ●   /
│       /   /  ← slope = θ₁
│      / ↗
│  θ₀ /   ← intercept (where line crosses y-axis)
│────/────────────────────── x
   0
```

- **θ₀ (intercept):** The value of `ŷ` when `x = 0`. It positions the line vertically.
- **θ₁ (slope):** For every 1-unit increase in `x`, `ŷ` changes by `θ₁` units.

### 📐 Closed-Form Formula for θ₀ and θ₁

Given `m` data points, the **optimal** θ₁ and θ₀ that minimize the sum of squared errors are:

**Step 1 — Compute means:**
```
x̄ = (1/m) · Σᵢ xᵢ
ȳ = (1/m) · Σᵢ yᵢ
```

**Step 2 — Compute θ₁ (slope):**
```
         Σᵢ (xᵢ - x̄)(yᵢ - ȳ)
θ₁  =  ─────────────────────────
              Σᵢ (xᵢ - x̄)²
```

> **Intuition:** The numerator measures how `x` and `y` vary *together* (covariance). The denominator measures how much `x` varies alone (variance). Their ratio gives the rate of change of `y` with respect to `x`.

**Step 3 — Compute θ₀ (intercept):**
```
θ₀ = ȳ - θ₁ · x̄
```

> **Intuition:** Once we know the slope, the intercept is determined by making the line pass through the centroid `(x̄, ȳ)` of the data.

### 🔢 Numerical Example

| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 4 |
| 5 | 5 |

```
x̄ = (1+2+3+4+5)/5 = 3
ȳ = (2+4+5+4+5)/5 = 4

Numerator   = (1-3)(2-4) + (2-3)(4-4) + (3-3)(5-4) + (4-3)(4-4) + (5-3)(5-4)
            = (-2)(-2) + (-1)(0) + (0)(1) + (1)(0) + (2)(1)
            = 4 + 0 + 0 + 0 + 2 = 6

Denominator = (-2)² + (-1)² + 0² + 1² + 2²
            = 4 + 1 + 0 + 1 + 4 = 10

θ₁ = 6/10 = 0.6
θ₀ = 4 - 0.6 × 3 = 4 - 1.8 = 2.2

→ Best fit line: ŷ = 2.2 + 0.6x
```

---

## 4. The Cost Function J(θ)

The **cost function** measures how wrong our predictions are across all training examples. We use **Mean Squared Error (MSE)**:

```
         1      m
J(θ) = ─────  · Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ]²
        2m     i=1
```

> The `1/2` factor is a **mathematical convenience** — it cancels with the `2` that appears when we differentiate, making the gradient cleaner. It does NOT affect where the minimum is.

### Why Squared Error?

- Penalizes **large errors more** than small ones (squared amplifies big mistakes)
- Always **non-negative** — easy to minimize
- **Differentiable everywhere** — enables gradient-based optimization
- Has a **unique global minimum** (convex for linear regression)

### Visualizing J(θ) — The Bowl Shape

For Simple Linear Regression, J(θ₀, θ₁) is a **convex paraboloid** in 3D:

```
J(θ)
 │\         /│
 │  \     /  │
 │    \   /  │
 │     \ /   │
 │      *    │  ← Global Minimum
 └───────────── θ₁
```

> Because J is **convex**, any local minimum IS the global minimum. Gradient descent is guaranteed to converge.

---

## 5. Math Intuition — Minimizing Cost

We want to find θ such that `J(θ)` is minimized. There are two approaches:

| Approach | Method | When to use |
|---|---|---|
| Iterative | Gradient Descent | Large datasets, many features |
| Closed-form | Normal Equation | Small datasets, few features |

---

## 6. Gradient Descent

**Gradient Descent** is an iterative optimization algorithm. It works by:

1. Starting at a random point on the cost surface
2. Computing the **slope (gradient)** at that point
3. Taking a small step in the **downhill direction** (negative gradient)
4. Repeating until convergence

### The Update Rule (Simultaneous Update)

```
Repeat until convergence {

    θⱼ  :=  θⱼ  −  α · ∂J(θ) / ∂θⱼ       for j = 0, 1, ..., n

}
```

Where:
- `:=` means **assignment** (update)
- `α` (alpha) = **learning rate** — controls step size
- `∂J(θ)/∂θⱼ` = partial derivative of cost w.r.t. θⱼ

> ⚠️ **CRITICAL:** Updates must be **simultaneous** — compute ALL new θⱼ values using the OLD θ values before updating any of them.

```python
# CORRECT — simultaneous update
temp0 = θ₀ - α * dJ/dθ₀
temp1 = θ₁ - α * dJ/dθ₁
θ₀ = temp0
θ₁ = temp1

# WRONG — sequential update (changes θ₀ first, then uses new θ₀ for θ₁)
θ₀ = θ₀ - α * dJ/dθ₀
θ₁ = θ₁ - α * dJ/dθ₁  # ← uses already-updated θ₀!
```

### Learning Rate α — The Critical Hyperparameter

```
Too small α:        Too large α:         Just right:
                    
J│╲                 J│  /\    /\         J│╲
 │  ╲               │ /  \  /  \          │  ╲
 │    ╲             │/    \/    \          │    ╲─
 │──────────        │────────────          │──────── (converges fast)
    (slow, many       (overshoots /
     iterations)       diverges)
```

**Rule of thumb:** Try `α` ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} and plot J vs iterations.

---

## 7. Deriving the Gradient Descent Update Rules

### Step-by-Step Partial Derivative Derivation

We need: `∂J(θ) / ∂θⱼ`

**Starting with the cost function:**
```
         1      m
J(θ) = ─────  · Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ]²
        2m     i=1
```

**Apply the chain rule:**
```
∂J(θ)     1    m         ∂
─────── = ───  Σ  2 · [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ] · ──  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ]
 ∂θⱼ     2m  i=1                              ∂θⱼ
```

**Simplify the 2:**
```
∂J(θ)     1    m
─────── = ───  Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ] · xⱼ⁽ⁱ⁾
 ∂θⱼ      m  i=1
```

> Because `∂h_θ/∂θⱼ = xⱼ` (derivative of `θ₀x₀ + θ₁x₁ + ...` with respect to `θⱼ` is just `xⱼ`)

---

### For θ₀ (j=0, x₀⁽ⁱ⁾ = 1 always):

```
∂J(θ)     1    m
─────── = ───  Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ]
 ∂θ₀      m  i=1
```

**Update rule for θ₀:**
```
            α    m
θ₀ := θ₀ − ─── · Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ]
             m  i=1
```

> **Intuition:** θ₀ is adjusted by the **average residual** across all training examples.

---

### For θ₁ (j=1):

```
∂J(θ)     1    m
─────── = ───  Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ] · x⁽ⁱ⁾
 ∂θ₁      m  i=1
```

**Update rule for θ₁:**
```
            α    m
θ₁ := θ₁ − ─── · Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ] · x⁽ⁱ⁾
             m  i=1
```

> **Intuition:** θ₁ is adjusted by the **weighted average residual** — the error weighted by how large `x` is, because bigger `x` values contributed more to the prediction error.

---

### Putting It All Together — Full Gradient Descent for Simple Linear Regression

```
Repeat until convergence {

    temp₀ := θ₀ − (α/m) · Σᵢ [ (θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾ ]

    temp₁ := θ₁ − (α/m) · Σᵢ [ (θ₀ + θ₁x⁽ⁱ⁾) - y⁽ⁱ⁾ ] · x⁽ⁱ⁾

    θ₀ := temp₀
    θ₁ := temp₁

}
```

---

### Vectorized Form (for Multiple Linear Regression)

```
θ := θ − (α/m) · Xᵀ · (Xθ - y)
```

Where:
- `X` is the design matrix of shape `(m × n+1)` — each row is one training example
- `y` is the target vector of shape `(m × 1)`
- `θ` is the parameter vector of shape `(n+1 × 1)`
- `Xθ - y` is the vector of residuals of shape `(m × 1)`

**Why vectorized?**
- No Python `for` loops → **100–1000× faster** with NumPy
- Works for any number of features `n`

---

## 8. The Normal Equation (Closed-Form Solution)

Instead of iterating, we can solve `∂J/∂θ = 0` directly:

```
θ = (XᵀX)⁻¹ · Xᵀ · y
```

### Derivation

Set the gradient to zero:
```
∂J/∂θ = (1/m) · Xᵀ(Xθ - y) = 0

→ XᵀXθ = Xᵀy

→ θ = (XᵀX)⁻¹ Xᵀy       (if XᵀX is invertible)
```

### Normal Equation vs. Gradient Descent

| Property | Normal Equation | Gradient Descent |
|---|---|---|
| Learning rate α | Not needed | Must be tuned |
| Iterations | None (single step) | Many iterations |
| Time complexity | O(n³) — matrix inversion | O(kn) per iteration |
| Best for | n < 10,000 features | n ≥ 10,000 features |
| Invertibility issue | (XᵀX) may be singular | No issue |

> If `XᵀX` is **non-invertible** (singular), it usually means: redundant features (multicollinearity) or more features than examples. Use pseudo-inverse `pinv()` or regularize.

---

## 9. Full Algorithm Summary

```
LINEAR REGRESSION — COMPLETE PIPELINE
═══════════════════════════════════════

INPUT: Training set { (x⁽ⁱ⁾, y⁽ⁱ⁾) }, learning rate α, iterations T

1. INITIALIZE:
   θ = zeros(n+1)         ← or small random values

2. FEATURE SCALING (recommended):
   x̃ⱼ = (xⱼ - μⱼ) / σⱼ   ← standardize each feature

3. ADD BIAS TERM:
   x₀⁽ⁱ⁾ = 1 for all i    ← prepend 1 to each example

4. FOR t = 1 to T:
   a. Compute predictions:  ŷ = Xθ
   b. Compute residuals:     e = ŷ - y
   c. Compute gradient:      ∇J = (1/m) · Xᵀe
   d. Update parameters:     θ ← θ - α · ∇J
   e. Record J(θ) for monitoring

5. OUTPUT: Learned parameters θ
```

### Convergence Check

```
If | J(θ)ₜ - J(θ)ₜ₋₁ | < ε   (e.g., ε = 10⁻⁶)
→ Stop: converged
```

Or: visually inspect J vs. iteration plot — should be **monotonically decreasing**.

---

## 10. Python Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression with Gradient Descent.
    Implements h_θ(x) = θ₀ + θ₁x₁ + ... + θₙxₙ
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.theta = None
        self.cost_history = []

    def _add_bias(self, X):
        """Prepend column of 1s for θ₀ (bias term)."""
        m = X.shape[0]
        return np.hstack([np.ones((m, 1)), X])

    def fit(self, X, y):
        """
        Train using Batch Gradient Descent.
        
        X: (m, n) — m examples, n features
        y: (m,)   — target values
        """
        X_b = self._add_bias(X)          # shape: (m, n+1)
        m, n = X_b.shape

        # Initialize θ to zeros
        self.theta = np.zeros(n)          # shape: (n+1,)

        for _ in range(self.n_iter):
            # Forward pass: compute predictions
            y_hat = X_b @ self.theta      # shape: (m,)

            # Compute residuals
            residuals = y_hat - y         # shape: (m,)

            # Compute cost J(θ) = (1/2m) Σ residuals²
            cost = (1 / (2 * m)) * np.sum(residuals ** 2)
            self.cost_history.append(cost)

            # Compute gradient: ∇J = (1/m) · Xᵀ · residuals
            gradient = (1 / m) * X_b.T @ residuals  # shape: (n+1,)

            # Update parameters: θ ← θ - α · ∇J
            self.theta -= self.lr * gradient

        return self

    def predict(self, X):
        """Predict using learned θ."""
        X_b = self._add_bias(X)
        return X_b @ self.theta

    def score(self, X, y):
        """Compute R² score."""
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_cost(self):
        """Plot cost function convergence."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_history, color='steelblue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('J(θ)')
        plt.title('Cost Function — Gradient Descent Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ─── Normal Equation (Closed-Form) ───────────────────────────────────────────

def normal_equation(X, y):
    """
    Computes optimal θ directly: θ = (XᵀX)⁻¹ · Xᵀ · y
    Uses np.linalg.pinv for numerical stability (handles singular matrices).
    """
    m = X.shape[0]
    X_b = np.hstack([np.ones((m, 1)), X])   # add bias
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y


# ─── Example Usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate synthetic data: y = 3 + 2x + noise
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1)
    y = 3 + 2 * X.squeeze() + np.random.randn(m)

    # Feature scaling (standardization)
    mu, sigma = X.mean(), X.std()
    X_scaled = (X - mu) / sigma

    # Train with Gradient Descent
    model = LinearRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X_scaled, y)

    print(f"θ (GD):            {model.theta}")
    print(f"R² Score:          {model.score(X_scaled, y):.4f}")
    print(f"Final Cost J(θ):   {model.cost_history[-1]:.6f}")

    # Compare with Normal Equation
    theta_ne = normal_equation(X_scaled, y)
    print(f"θ (Normal Eq.):    {theta_ne}")

    model.plot_cost()
```

---

## 11. Key Hyperparameters & Pitfalls

### Feature Scaling (Always Do This!)

Gradient Descent converges **much faster** when features are on similar scales.

**Standardization (Z-score):**
```
x̃ = (x - μ) / σ
```

**Min-Max Normalization:**
```
x̃ = (x - xₘᵢₙ) / (xₘₐₓ - xₘᵢₙ)   → scales to [0, 1]
```

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| α too large | J oscillates or diverges | Reduce α by 10× |
| α too small | J decreases very slowly | Increase α or use adaptive methods |
| No feature scaling | Skewed contours, slow convergence | Always standardize |
| Not enough iterations | θ hasn't converged | Increase `n_iterations` |
| Multicollinearity | (XᵀX) near-singular | Use `pinv` or regularize |
| Overfitting | High variance on test set | Add L2 regularization (Ridge) |

### Regularization Extension — Ridge Regression (L2)

To prevent overfitting, add a penalty on θ magnitude:

```
         1    m                          λ    n
J(θ) = ─────  Σ  [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ]²  + ─── · Σ  θⱼ²
        2m   i=1                         2m   j=1
```

> Note: θ₀ (bias) is typically **NOT** regularized.

The gradient update becomes:
```
θⱼ := θⱼ · (1 - α·λ/m)  −  (α/m) · Σᵢ [ h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾ ] · xⱼ⁽ⁱ⁾
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                 LINEAR REGRESSION CHEAT SHEET               │
├──────────────────────┬──────────────────────────────────────┤
│ Hypothesis           │ h_θ(x) = θᵀx                        │
│ Cost Function        │ J(θ) = (1/2m)·‖Xθ - y‖²            │
│ Gradient             │ ∇J = (1/m)·Xᵀ·(Xθ - y)             │
│ Update Rule          │ θ ← θ - α·∇J                        │
│ Normal Equation      │ θ = (XᵀX)⁻¹·Xᵀ·y                   │
│ θ₁ (slope)          │ Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²         │
│ θ₀ (intercept)      │ ȳ - θ₁·x̄                            │
│ R² Score             │ 1 - SS_res/SS_tot                    │
└──────────────────────┴──────────────────────────────────────┘
```

---

*Notes compiled for deep learning & ML study. Math derivations follow Andrew Ng's ML course conventions.*