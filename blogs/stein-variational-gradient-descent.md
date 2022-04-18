---
description: A new way to learn
---

# Stein Variational Gradient Descent

## 

## Introduction

Gradient descent has become the basic algorithm for the training of almost all deep learning models. Stein variational gradient descent was proposed as a "natural counterpart of gradient descent for optimization."

In our previous blog post *Differentiable Bayesian Structure Learning*, we briefly mentioned the core engine of the Bayesian algorithm was the Stein variational gradient descent. In this article, we will expand this topic and articulate the motivation, fundamental logic, mathematical derivation of this novel optimization method. The original paper can be found in the Reference section while in this article we will derive the algorithm in a much more intuitive way.

## Notations

The usual inner product of two functions $f$ and $g$ is defined as

$$
\langle f, g \rangle = \int dx \; f(x)g(x)
$$

Let $$f$$ and $$g$$ be functions in [Reproducing kernel Hilbert space](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) (RKHS) $$\mathcal H$$, then the inner product is denoted $$\langle f, g \rangle_\mathcal H$$ .

Let $$\{f_\mu\}_{\mu=1}^d$$ and $$\{g_\nu\}_{\nu=1}^d$$ be functions in RKHS $$\mathcal H^d$$, then the inner product 

$$
\langle \mathbf f, \mathbf g \rangle_{\mathcal H^d} = \delta_{\mu\nu}\langle f_\mu, g_\nu \rangle_{\mathcal H^d}.
$$

## Stein identity and discrepancy

Observe that using Stokes' Theorem

$$
\int_{M} f\mathrm dp + p\mathrm df = \int_{M} \mathrm d (pf) = \int_{\partial M} pf
$$

if $$pf \rightarrow 0$$ at boundary $$\partial M$$ and both $p$ and $$f$$ are smooth functions, then we have

$$
\int_M f\mathrm d p + p \mathrm df = 0.
$$

 If $$p$$ is a probability density over $$M$$, then we have the Stein **identity**

$$
\int_M fp\;  \mathrm d\log p + p\, \mathrm d f 
= \mathbb E_p \left[ f\,  \mathrm d \log p + \mathrm d f \right] = 0
$$

for any test function $f$ that satisfies the requirements. Now we replace the sampling distribution by $$q$$ 

$$
\mathbb E_p [f \mathrm d\log p + \mathrm d f] \rightarrow \mathbb E_q [f \mathrm d\log p + \mathrm d f]\equiv S_{q,p}f
$$

we get the Stein discrepancy which vanishes when $$q$$ is $$p$$. Thus, we obtain a measure of "distance" between $$q$$ and $$p$$ with properly chosen test function $$f$$.

## Variational inference

The goal of variational inference is to approximate a target distribution $$p$$ with a tractable ansatz distribution $$q$$ by minimizing the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

$$
D_{\rm KL} (q||p) = \mathbb E_q \log \frac{q}{p} = (-\mathbb E_q\log p) - (-\mathbb E_q \log q)
$$

which is in the form of free energy $$F = U - TS$$ with temperature equals unity. Thus, minimization of KL-divergence is equivalent to striking a balance between minimizing $$q$$-average of energy ($$-\log p$$) and maximizing the entropy of $$q$$.

KL-divergence is non-negative and attains minimum zero when $$q=p$$. We wish to $$\min_q D_{\rm KL}(q||p)$$ subject to the constraint that $q$ is a probability distribution $$\int dx\, q(x) = \langle q, 1 \rangle = 1$$. Thus, the total objective is

$$
\mathcal L[q] = \langle q, \log q - \log p \rangle - \lambda \langle q, 1 \rangle
$$

where $$\lambda$$ is a Lagrange multiplier. Take functional derivative

$$
\frac{\delta \mathcal L}{\delta q} = \log q - \log p + 1 - \lambda = 0 
$$

we get

$$
q = p \exp (\lambda -1).
$$

Since $$p$$ is a distribution, then we get $$\lambda=1$$. Thus, we showed $$q=p$$ is the solution of the optimization problem. However, for a real-world distribution $$p$$, which may be arbitrarily complicated, it is impossible to obtain an exact equality but a best approximation given the functional form of ansatz $$q$$.

The ansatz distribution can be manually constructed based on the knowledge of the target distribution, e.g. mean-field approximation using exponential family distributions. In this article, we will discuss a non-parametric approach using particles of $$q$$.

## Coordinate flow

Since both $$q$$ and $$p$$ are smooth functions and more importantly probability distributions, we can adiabatically deform $$q$$ into $$p$$ by shifting the coordinate

$$
x^\mu  \mapsto x^\mu + v^\mu(x) \delta t
$$

where for simplicity we assume $$x \in \mathbb R^d$$.

The task of seeking such a transformation is equivalent to searching for a proper velocity field $$v(x)$$. 

The total mass of $$q$$ is conserved and we have a conserved current of $$q$$-charge

$$
j_q^\mu (x) = q(x) v^\mu(x)
$$

and

$$
\dot q (t) = - \partial_\mu j^\mu _q = - v^\mu \partial_\mu q - q \partial_\mu v^\mu .
$$

We have the following equivalent optimization problems

$$
\min_{v} D_{\rm KL} (q_{t+\delta t}||p_t)) 
\Leftrightarrow 
\min_{v} D_{\rm KL}(q_t||p_{t - \delta t})
$$

which is the equivalence of active and passive perspectives of coordinate transformations. In the latter case, we have the velocity in reverse direction

$$
\dot p (t) = - \partial_\mu j_p^\mu = v^\mu \partial_\mu p + p \partial_\mu v^\mu . 
$$

where $$j_p$$ is the conserved current of $$p$$-charge.

We have the following expansion

$$
p(x, t-\delta t) = p(x, t) - \dot p (x, t) \delta t + \mathcal O(\delta t^2)
$$

$$
\log p(x, t-\delta t) = \log p(x, t) - \frac{d}{dt}\log p (x, t) \, \delta t + \mathcal O(\delta t^2) .
$$

The increment of the objective 

$$
\mathcal L[v] 
= D_{\rm KL}(q_t||p_{t - \delta t}) 
= \int dx\, q_t(x) \log \frac{q_t(x)}{p_{t-\delta t}(x)}
$$

in first order of $$\delta t$$

$$
\delta \mathcal L[v] = - \delta t \int dx\, q_t(x) \frac{\dot p_t(x)}{p_t(x)} .
$$

Replace $$\dot p$$ using the continuity equation, we get

$$
\frac{ \delta \mathcal L[v] }{\delta t}
= - \int dx\, q(x) (v^\mu(x) \partial_\mu p(x) + \partial_\mu v^\mu) 
= - \mathbb E_q(v^\mu \partial_\mu p + \partial_\mu v^\mu).
$$

In other words, the gradient of $$D_{\rm KL}(q_t||p_{t - \delta t})$$ is the negative Stein discrepancy of $$(q,p)$$ using test function $$v$$, i.e. $$S_{q,p} v$$.

## Method of steepest descent

Now we got the gradient descent of our variational inference objective, but we wish to get the steepest descent by searching for a proper velocity field.

We further assume the velocity field is an element of $$d$$-dimensional RKHS $$\mathcal H^d$$. Then we have the reproducing property

$$
v = \langle K, v \rangle_{\mathcal H^d} = \langle v, K \rangle_{\mathcal H^d} 
$$

where $$K(\cdot, \cdot)$$ is the kernel function of $$\mathcal H^d$$, e.g. Gaussian RBF kernel. Furthermore, the linear operator $$S_{q,p}$$ can be shifted to the $$K$$,

$$
S_{q,p}v = \langle v, S_{qp}K \rangle_{\mathcal H^d}
$$

which is the "dot product" of $$v$$ and $$S_{qp} K$$ in RKHS. 

The solution of the optimization

$$
\max_{v, \Vert v \Vert_{\mathcal H^d} \le 1} S_{qp} v = \langle v, S_{qp}K \rangle_{\mathcal H^d}
$$

is simply $$v^* = S_{qp}K/\Vert S_{qp}K \Vert_{\mathcal H^d}$$. Also note that the velocity will vanish when $$p=q$$.

## The Stein variational inference algorithm

Using method of steepest descent, we obtained the optimal flow field $$v^*$$. Next, we just need to go with flow

$$
x \mapsto x + v^* \delta t = x + \delta t'\, S_{qp}K 
$$

and incrementally update $$q(x)$$. 

The algorithm is as follows:

- Sample $$m$$ particles of initial $$q$$

- Approximate $$S_{qp}K = \mathbb E_q [K^\mu \partial_\mu \log p + \partial_\mu K^\mu]$$ using sample mean of $$m$$ particles

- The inverse of the norm of $$v^*$$ is absorbed into learning rate $$\delta t'$$

- Update the coordinates of the particles using the calculated flow

- Repeat the process



## Reference

1. [[1608.04471] Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471)
