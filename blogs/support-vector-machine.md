---
description: Formulation of Support Vector Machine (SVM)
---

# Support Vector Machine

### Primal Optimization Problem

SVM is a maximum margin classifier where the decision boundary $$f(x) = \vec w\cdot \vec x + \vec b$$ separate two classes labeled by $$y=\pm 1$$ in $$\mathbb R^n$$ with a largest possible margin.

#### The normal vector of the hyperplane

The normal vector in direction of increasing $$f(x)$$ is

$$
\hat n = \frac{\nabla f(x)}{\Vert \nabla f(x) \Vert_2} = \frac{\vec w}{\Vert w \Vert_2}
$$

#### The perpendicular distance to the hyperplane from any point $$\vec x'\in\mathbb R^n$$ &#x20;

The distance between any point $$x'$$ and the plane $$f(x)$$ is the distance between this point and any point in the plane projected to the normal direction $$d = \vert (\vec x' - \vec x)\cdot \hat n \vert$$ where $$f(x) = \Vert w \Vert \hat n\cdot \vec x + b$$. Now ground the decision boundary to zero potential, then $$0 = \Vert w \Vert \hat n\cdot \vec x + b$$ and thus,

$$
d=\frac{\left\vert \vec w\cdot \vec x' + b  \right\vert}{\Vert w \Vert}
$$

#### Maximize the distance subject to the constraint of correct class labels

The optimization is

$$
\max_{w, b} \min_x d = \max_{w, b}\min_{x} \frac{\left\vert \vec w\cdot \vec x +  b  \right\vert}{\Vert w \Vert}
$$

subject to

$$
\vec w \cdot \vec x +  b \ge 1,\, y=+1\\
\vec w \cdot \vec x  +  b \le -1,\, y=-1
$$

which is equivalent to

$$
\max_{w,b} \frac{1}{\Vert w \Vert} = \min_{w,b} \frac{1}{2}\Vert w \Vert_2^2,\; \forall (\vec x,y) \; 1 - y(\vec w \cdot \vec x + b) \le 0
$$

Introduce KKT parameters $$\alpha_i$$ for each pair of data and label $$(\vec x_i, y_i)$$, the primal Lagrangian, or cost function is

$$
\mathcal L^P(w,b) = \frac12 w^\top w + \sum_i \alpha_i [1-y_i(w^\top x_i + b)]
$$

with

$$
\alpha_i [1 - y_i(w^\top x_i +b)] = 0, \alpha_i > 0
$$

#### Soft margin and slack variables

The hinge loss $$\max(0, 1-y(w^\top x + b))$$ measures the degree of misclassification where slack variable $$\xi = 1 - y(w^\top x + b)$$.

The optimization becomes

$$
\min_{w,b} \frac12 \Vert w \Vert^2 + C\sum_i \max(0, 1-y_i(w^\top x_i  + b))
$$

If we wish to use the dual form, we need to cast the above into constraints. Now the soft margin constraint becomes

$$
(1-\xi) - y(w^\top x + b) \le 0,\, \xi \ge 0
$$

The primal Lagrangian becomes

$$
\mathcal L^P(w,b, \{\xi_i\}) = \frac12 w^\top w + C\sum_i \xi_i + \sum_i \alpha_i [(1-\xi_i)-y_i(w^\top x_i + b)] - \sum_i \mu_i\xi_i
$$

The gradients are

$$
\partial_w \mathcal L^P = w - \sum_i \alpha_i y_i x_i=0\\
\partial_b \mathcal L^P = -\sum_i \alpha_i y_i = 0\\
\partial_{\xi_i} \mathcal L^P = C - \mu_i - \alpha_i=0
$$

while $$\mu_i > 0$$, $$\alpha_i >0$$, $$\xi_i \ge 0$$.

### Dual Formulation

The dual Lagrangian reads

$$
\mathcal L^D = \sum_i \alpha_i -\frac12\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle
$$

where $$C-\mu$$ gives $$\alpha$$ and hinge loss gives $$-w^2$$ combine the first term gives $$-\frac12 w^2$$.

Now the optimization problem becomes

$$
\max_{\{\alpha_i\}} \sum_i \alpha_i -\frac12\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle
$$

subject to $$\alpha_i\in[0, C]$$.

#### Kernel Trick

Replace $$\langle x_i, x_j \rangle$$ with a kernel function $$K(x_i,x_j) = \langle \phi(x_i),\phi(x_j)\rangle_K$$. The weight becomes (Representer Theorem)

$$
w = \sum_i \alpha_i y_i \phi(x_i)
$$

and the decision hypersurface

$$
f(x) = \langle w, \phi(x) \rangle + b = \sum_i \alpha_i y_i \langle \phi(x_i), \phi(x)\rangle + b = \sum_i \alpha_i y_i K(x_i, x) + b
$$

where the bias (or threshold) can be solved using

$$
y_j f(x_j) = 1 \Rightarrow \sum_{i} \alpha_i y_i y_j K(x_i, x_j) = 1 - b y_j
$$

where $$\phi(x_j)$$ is any support vector with $$\alpha_j > 0$$
