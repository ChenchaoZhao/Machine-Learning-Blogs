---
description: >-
  An algorithm for novelty detection and empirical distribution support
  estimation
---

# One Class SVM

#### Separate data distribution from the origin by a hyperplane

Assume the points are separable from the origin, then what is the maximum margin?

The problem can be formulated as

$$
\min_{w} \frac12 \Vert w \Vert^2 \;\text{ subject to }\;(w\cdot x_i) \ge \rho, \text{ with }\rho\ge 0
$$

for all data point $$x_i \in \mathbb R^n$$. The problem is equivalent to the binary case where $$(x_i; 1)$$ and $$(-x_i; -1)$$ are the two classes.

The Lagrangian is

$$
\mathcal L(w) = \frac12 w^2 - \sum_i \alpha_i (w\cdot x_i - \rho) - \mu \rho
$$

with KKT conditions $$\alpha_i (w\cdot x_i -\rho) = 0$$ where $$\alpha_i \ge 0$$.

Take gradient

$$
\partial_w \mathcal L = w - \sum_i \alpha_i x_i = 0\\
\partial_\rho \mathcal L = \sum_i \alpha_i - \mu = 0
$$

Then, $$w=\sum_i \alpha_i x_i$$. The dual problem Lagrangian is

$$
\mathcal L^* (\alpha) = - \frac12 \sum_{i,j} \alpha_i \alpha_j \langle x_i, x_j\rangle \text{ with }\alpha_i \ge 0 \text{ and } \sum_i\alpha_i = {\rm const.} \ge 0
$$

Without loss of generality, we set $$\sum_i \alpha_i = 1$$.

In other words,

$$
\max_{\alpha}\;  - \frac12 \alpha^\top K \alpha \;\text{ subject to } \alpha_i\in[0,\infty), \alpha^\top 1 = 1
$$

or

$$
\min_{\alpha} \frac12 \alpha^\top K \alpha \;\text{ subject to } \alpha_i\in[0,\infty), \alpha^\top 1 = 1
$$

The $$\rho$$ can be solved by plugging in the support vectors, $$\rho = w\cdot x_i^*$$ .&#x20;

The decision function is given by $$f(x) = {\rm sgn}(w\cdot x -\rho)$$ and $$f(x_i)\ge 0$$ for all data points.

#### Inseparable case and soft margin

Now we allow the violations of classification criterion $$w\cdot x_i \ge \rho$$, and panelize the misclassifications using Hinge loss

$$
\mathcal L^{\rm hinge} (w, \rho) = \sum_i \max(0, -(w\cdot x_i-\rho)) = -\sum_i (w\cdot x_i -\rho) 1_{\{w\cdot x_i < \rho\}}
$$

Introduce slack variable $$\xi_i \ge 0$$ as the discrepancy $$\xi_i = \rho - w \cdot x_i$$ for each violation. With help of slack variables, we have

$$
w\cdot x_i -\rho +\xi_i \ge 0
$$

Thus, the soft margin loss function

$$
\mathcal L(w, \rho, \xi) = \frac12 \Vert w\Vert^2 + C\sum_i \xi_i - \sum_i \alpha_i (w\cdot x_i -\rho +\xi_i) - \sum_i \beta_i \xi_i - \mu\rho
$$

where $$C\ge0$$ and $$\mu\ge 0$$. The gradients are

$$
\partial_w \mathcal L = w - \sum_i \alpha_i x_i = 0\\
\partial_\rho \mathcal L = \sum_i \alpha_i - \mu = 0\\
\partial_{\xi_j} \mathcal L = C - \alpha_j - \beta_j = 0
$$

Thus, the dual problem is

$$
\mathcal L^*(\alpha) = -\frac12 \sum_{i,j}\alpha_i\alpha_j \langle x_i,x_j \rangle = -\frac12 \alpha^\top K \alpha
$$

Hence we have the quadratic program

$$
\min_{\alpha} \frac12 \alpha^\top K \alpha \;\text{ subject to }\, \alpha_i \in [0, C] \,\text{ and } \sum_i \alpha_i = 1
$$

where $$C=(\nu m_{\rm sample})^{-1}$$ in the original paper and thus $$C^{-1}$$ is the upper bound of number of outliers.
