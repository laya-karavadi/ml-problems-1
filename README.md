# Machine Learning Problems: Gradient Descent, Polynomial Regression, and Lasso Regularization

This repository contains solutions to three fundamental machine learning problems demonstrating optimization techniques and regularization methods.

## Problem 1: Gradient Descent Optimization

### Overview
Implements gradient descent algorithm to minimize a quadratic loss function with two parameters.

### Problem Setup
- **Loss function:** `J(θ₁, θ₂) = θ₁² + 4θ₁θ₂ + θ₂²`
- **Initial values:** `θ₁ = -3, θ₂ = 3`
- **Learning rates tested:** `α = 0.03` and `α = 0.5`

### Gradient Calculations
The partial derivatives of the loss function are:
- `∂J/∂θ₁ = 2θ₁ + 4θ₂`
- `∂J/∂θ₂ = 4θ₁ + 2θ₂`

At initial point `(-3, 3)`:
- `∂J/∂θ₁ = 2(-3) + 4(3) = 6`
- `∂J/∂θ₂ = 4(-3) + 2(3) = -6`

### Parameter Updates
Using the gradient descent update rule: `θ* = θ - α * ∇J`

**With α = 0.03:**
- `θ₁* = -3 - 0.03 × 6 = -3.18`
- `θ₂* = 3 - 0.03 × (-6) = 3.18`

**With α = 0.5:**
- `θ₁* = -3 - 0.5 × 6 = -6`
- `θ₂* = 3 - 0.5 × (-6) = 6`

### Results Comparison
| Learning Rate | Initial Loss | Updated Loss | New Parameters |
|---------------|--------------|--------------|----------------|
| Initial       | -18          | -            | (-3, 3)        |
| α = 0.03      | -18          | -20.10       | (-3.18, 3.18)  |
| α = 0.5       | -18          | -72          | (-6, 6)        |

### Key Insights
- The larger learning rate (α = 0.5) produces much larger parameter updates
- Higher learning rates can lead to overshooting and potentially unstable optimization
- Smaller learning rates provide more controlled convergence

---

## Problem 2: Polynomial Regression with L2 Regularization

### Overview
Implements polynomial regression with degree 3 features and applies L2 (Ridge) regularization to prevent overfitting.

### Dataset Generation
```python
# Generate 10 samples with 2 features
X = np.random.rand(10, 2)

# Create target with cubic relationship
y = 4 * X[:, 0]**3 + 3 * X[:, 1]**3 + 2 * X[:, 0] * X[:, 1] + noise
```

### Feature Engineering
The polynomial feature expansion creates combinations up to degree 3:
- **Degree 0:** Bias term (1)
- **Degree 1:** x₁, x₂
- **Degree 2:** x₁², x₁x₂, x₂²
- **Degree 3:** x₁³, x₁²x₂, x₁x₂², x₂³

### Regularization
Applied L2 regularization with λ = 0.1:
```python
# Ridge regression solution
w = (X_poly.T @ X_poly + λ * I)⁻¹ @ X_poly.T @ y
```

### Model Output
The learned polynomial equation takes the form:
```
y = w₀ + w₁*x₁ + w₂*x₂ + w₃*x₁² + w₄*x₁*x₂ + w₅*x₂² + w₆*x₁³ + w₇*x₁²*x₂ + w₈*x₁*x₂² + w₉*x₂³
```

---

## Problem 3: Lasso Regression (L1 Regularization)

### Overview
Implements Lasso regression using gradient descent with soft thresholding to achieve sparse solutions.

### Algorithm Implementation
```python
def lasso_regression(X, y, lambda_val, learning_rate=0.001, epochs=2000):
    # Gradient descent with L1 penalty
    # Soft thresholding: sign(w) * max(0, |w| - λα)
```

### Regularization Analysis
Tested three different λ values to observe sparsity effects:

| Lambda (λ) | Zero Coefficients Ratio | Sparsity Effect |
|------------|-------------------------|-----------------|
| 0.01       | Low                     | Minimal sparsity |
| 0.1        | Medium                  | Moderate sparsity |
| 1.0        | High                    | Strong sparsity |

### Key Findings
- **Sparsity Induction:** As λ increases, more coefficients become exactly zero
- **Feature Selection:** L1 regularization automatically performs feature selection
- **Model Interpretability:** Sparse models are easier to interpret and less prone to overfitting

---

## Technical Requirements

### Dependencies
```python
import numpy as np
from itertools import combinations_with_replacement
```

### Key Concepts Demonstrated
1. **Gradient Descent:** First-order optimization algorithm
2. **Learning Rate Impact:** Effect of step size on convergence
3. **Polynomial Features:** Non-linear feature engineering
4. **L2 Regularization (Ridge):** Prevents overfitting by penalizing large weights
5. **L1 Regularization (Lasso):** Promotes sparsity and feature selection
6. **Soft Thresholding:** Key technique for L1 optimization

### Mathematical Foundations
- **Ridge Penalty:** `λ∑wᵢ²`
- **Lasso Penalty:** `λ∑|wᵢ|`
- **Soft Thresholding:** `sign(z) * max(0, |z| - λ)`

---

## Usage Notes
1. Set random seed for reproducible results
2. Experiment with different learning rates and regularization parameters
3. Monitor convergence behavior and adjust hyperparameters accordingly
4. Compare regularization methods based on your specific requirements (sparsity vs. stability)

## Further Extensions
- Implement elastic net (combination of L1 and L2)
- Add cross-validation for hyperparameter tuning
- Visualize loss landscapes and convergence paths
- Compare with scikit-learn implementations
