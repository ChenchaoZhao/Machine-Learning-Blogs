---
description: Heat kernel on high-dimensional spheres
---

# Spherical counterpart of Gaussian kernel

## Introduction

Normal distribution may be the most common and important continuous distribution. Gaussian radial basis kernel (RBF) is also widely applied in various kernel methods such as SVM and, kernel PCA. Gaussian kernel is essentially the same as normal density up to a scaling factor such that self-similarity is one.

Cosine similarity is often employed to measure overlaps between high-dimensional feature vectors where vectors in $$\mathbb R^n / \{0\}$$are normalized by their positive $$\ell_2$$-norms. In this case, the feature or embedding vectors are identified as points on a unit $$S^{n-1}$$.

## Generalization of Gaussian kernel

If the feature space is a high-dimensional **sphere**, what is the **equivalent of Gaussian RBF** kernel?

### von Mises-Fisher

The [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von\_Mises%E2%80%93Fisher\_distribution) is directly related to normal distribution by restricting the density to a unit hypersphere. The corresponding kernel is $$\sim \exp (\kappa \cos \theta)$$ where $$\kappa > 0$$ is the precision parameter. If we rescale the kernel such that self-similarity is one, then we have

$$
K_\text{vmf}(\hat x, \hat y) = {\rm e}^{\kappa (\cos \theta - 1)} = {\rm e}^{ \kappa ( \hat x \cdot \hat y - 1)} = {\rm e}^{- \frac{\kappa}{2} \Vert \hat x - \hat y \Vert^2 }
$$

which is Gaussian RBF kernel in disguise with Euclidean distance as the chordal length $$\Vert \hat x - \hat y \Vert$$.

### Geodesic Gaussian

If we replace the Euclidean distance squared by geodesic distance squared, we have a measure of similarity $$\sim \exp (- \kappa \theta^2 )$$. However, the symmetric binary function $$S^{n-1} \times S^{n-1} \rightarrow \mathbb R$$ is **not positive definite** which can be proved using randomly generated examples. Thus, the geodesic Gaussian function is not a reproducing kernel. Hence, it cannot be applied to any of the kernel methods based on the assumption of reproducing kernel Hilbert space (RKHS).

### Heat Kernel

#### Euclidean heat kernel

Normal distribution is the Green's function of heat (diffusion) equation in Euclidean space. If we assume the diffusion constant is isotropic and equals one, then we get a one-parameter family of kernels parameterized by diffusion time $$t$$,

$$
G_\text{euc} (\Vert x - y \Vert; t; n) = \left ( \frac{1}{4\pi t }\right )^{\frac{n}{2}} \exp \left( - \frac{\Vert x - y \Vert^2}{4t}\right)
$$

#### Spherical heat kernel

The Laplacian operator in Euclidean space may be interpreted as momentum squared or kinetic energy. The spherical Laplacian operator must be the kinetic energy of a particle on a hypersphere and the kinetic energy is quantum mechanical angular momentum squared. For more details, please refer to [Exact Heat Kernel on a Hypersphere and Its Applications in Kernel SVM](https://www.frontiersin.org/articles/10.3389/fams.2018.00001/full). The heat kernel on $$S^{n-1}$$ takes the exact expansion

$$
G_\text{sph} (\hat x \cdot \hat y; t; n-1) = \sum_{\ell = 0}^\infty {\rm e}^{-\ell (\ell + n - 2) t}  \frac{2\ell + n - 2}{n - 2} \frac{1}{A_{S^{n-1}}} C_{\ell}^{\frac{n}{2} - 1}(\hat x \cdot \hat y)
$$

where $$A_{S^{n-1}}$$ is area of $$S^{n-1}$$ and $$C_\ell^\alpha (w)$$ are Gegenbauer polynomials.

For very large time, the heat must be uniform on the sphere. It is easy to verify that when $$t\uparrow \infty$$ the heat kernel reduces to uniform distribution $$1/A_{S^{n-1}}$$.&#x20;

The kernel was obtained through eigenfunction expansion with positive eigenvalues. Thus, it is not only positive definite but also a [Mercer kernel](https://en.wikipedia.org/wiki/Mercer's\_theorem).

If we rescale the kernel such that self-similarity is unity, we have

$$
K_\text{sph}(\hat x \cdot \hat y; t) = G_\text{sph}(\hat x \cdot \hat y; t)/G_\text{sph}(1; t).
$$

#### Reduction to Euclidean heat kernel within a small vicinity

When $$\hat x \cdot \hat y \equiv \cos \theta \approx 1$$, the hypersphere is locally homeomorphic to $$\mathbb R^{n-1}$$and the physics of heat diffusion should reduce to that of Euclidean space as well. Therefore, we need to show that the exact hyperspherical heat kernel reduces to Euclidean heat kernel for $$\theta \downarrow 0$$.&#x20;

First, let $$\hat x \cdot \hat y = 1 - \varepsilon$$ then&#x20;

$$
G_\text{sph} (1 - \varepsilon; t) = \sum_{\ell = 0}^\infty {\rm e}^{-\ell (\ell + n - 2) t}  \frac{2\ell + n - 2}{n - 2} \frac{1}{A_{S^{n-1}}} C_{\ell}^{\frac{n}{2} - 1}(1 - \varepsilon).
$$

Gegenbauer polynomials can be expressed using hypergeometric functions

$$
C_{\ell}^{\alpha}(x) = \binom{\ell + 2\alpha - 1}{\ell}\,_2F_1\left(-\ell, \ell + 2\alpha; \alpha + \frac12; \frac{1-x}{2} \right).
$$

Keep only first order terms, and let$$x=1-\varepsilon$$and $$\alpha = \frac{n}{2} - 1$$&#x20;

$$
C_{\ell}^{\frac{n}{2} - 1}(1 - \varepsilon) = \binom{\ell + 2n - 3}{\ell}\left( 1 - \frac{\ell (\ell + n - 2)}{n-1}\varepsilon + \mathcal O (\varepsilon^2)\right).
$$

The heat kernel becomes

$$
G_\text{sph} (1 - \varepsilon; t) = G_\text{sph} (1; t)+ \frac{1}{n-1}\partial_t G_\text{sph} (1; t) \varepsilon + \mathcal O(\varepsilon^2) \approx G_\text{sph} (1; t) {\rm e}^{\frac{1}{n-1} \partial_t \log G_\text{sph} (1; t) \varepsilon}
$$

Set $$\varepsilon = \theta^2 / 2$$ and $$f(t) = 1 / G_\text{sph}(1; t)$$. In geodesic spherical coordinates, normalization of the density requires

$$
f(t) = \left (2\pi \frac{n-1}{\partial_t \log f(t)} \right)^{\frac{n-1}{2}}
$$

&#x20;which is a nonlinear differential equation for $$f(t)$$. Let $$f(t) = C t^\alpha$$, then we have the equality

$$
\frac{C^\frac{2}{n-1} \alpha}{2\pi (n-1)} = t^{1 - \frac{2 \alpha}{n-1}}=1
$$

&#x20;which gives $$\alpha = (n-1)/2$$ and $$C=(4\pi)^{(n-1)/2}$$ or $$f(t) = (4\pi t)^{(n-1)/2}$$.

Thus, for small geodesic distance, we have

$$
G_\text{sph}(\cos\theta; t; n-1) = G_\text{euc}(\theta; t; n-1) + \mathcal O (\theta^4).
$$

Hence, we have shown that the spherical heat kernel is indeed an extension of Euclidean heat kernel to the whole hypersphere.
