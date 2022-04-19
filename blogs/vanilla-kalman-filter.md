---
description: BS-free introduction to the math of Kalman filter
---

# Vanilla Kalman Filter



## Graphical model

![Hidden states are denoted as "x" and observables as "z" ](../.gitbook/assets/kalman\_graph.svg)

## Kindergarten physics

$$
{\bf r}(t+\delta t) = {\bf r}(t) + \dot {\bf r}(t) \delta t + \frac12 \ddot {\bf r}(t) \delta t^2 + {\rm h.o.t}
$$

If assume $$\langle \ddot {\bf r} \rangle = 0$$, then the state $$x$$ is characterized by $$({\bf r}, \dot {\bf r})$$, i.e.

$$
x_{i}(t+\delta t) = x_i (t) + v_i (t) \delta t +\frac12 \varepsilon_i(t) \delta t^2\\
v_{i}(t+\delta t) = v_i(t) + \varepsilon_i (t) \delta t \qquad\qquad\qquad
$$

Time evolution

$$
x_a(t+\delta t) = F_{ab} x_b(t) + \eta_a
$$

$$
F_{ab} = \pmatrix{
1 & \delta t \\
0 & 1
}\quad \eta_a \sim {\cal N}(0, {Q})
$$

Observation or measurement

$$
z_a(t) = H_{ab} x_b(t) + \xi_a\quad \\
H_{ab} = \pmatrix{
1 & 0 \\
}
\quad \xi_a\sim {\cal N}(0, {R})
$$

## Inference

* Given observations $$z_{<t}$$ up to $$t-1$$, what is the distribution of state $$x_t$$?

$$
p(x_t|z_{<t}) = \int {\rm d}x_{t-1}\; p(x_t | x_{t-1}) p (x_{t-1} | z_{\le t-1})
$$

* If there is one more observation $$z_t$$, what is the distribution of state $$x_t$$?

$$
p(x_t|z_{\le t}) = p(x_t | z_t, z_{<t}) \propto p(z_t | x_t) p(x_t | z_{<t})
$$

Expand the recursion, we have Feynman-Kac formula or path integral

$$
p(x_t | z_{< t}) \propto \int \left [ \prod_{\tau = 1}^{t-1} {\rm d}x_\tau \right]\, \prod _{\tau=1}^{t}p(z_\tau | x_\tau) p(x_\tau | x_{\tau - 1})  p(z_0 | x_0) p(x_0)
$$

If we assume the priors, transition probabilities and emission probabilities are all normal, then the posterior of state $$x_t$$ is also normal.

Thus, we assume

$$
p(x_t | z_{< t}) = \mathcal N (\hat x_{t|t-1}, {P}_{t|t-1})\\
p(x_t | z_{\le t}) = \mathcal N (\hat x_{t|t}, {P}_{t|t})\qquad
$$

### State means

Use the recursion again,

$$
p(x_t | z_{\le t}) \propto p(z_t | x_t) p(x_t | z_{<t})
$$

$$
\mathcal N(x_t; \hat x_{t|t}, P_{t|t}) \propto \mathcal N(z_t; Hx_t, R) \mathcal N(x_t; \hat x_{t|t-1}, P_{t|t-1})
$$

Maximize $$x_t$$ __ on both sides, the LHS __ $$x_t = \hat x _{t|t}$$, the RHS

$$
{\rm rhs}=(z_t - Hx_t|R^{-1}|z_t - Hx_t) + (x_t - \hat x_{t|t-1}|P_{t|t-1}^{-1}|x_t - \hat x_{t|t-1})
$$

$$
0=\partial_{x_t^\top}{\rm rhs}= - H^\top R^{-1}(z_t - Hx_t) + P^{-1}_{t|t-1} (x_t - \hat x_{t|t-1})
$$

We must have

$$
\begin{align}
(\hat x_{t|t} - \hat x_{t|t-1}) 
&= P_{t|t-1} H^\top R^{-1} (z_t - H \hat x_{t|t})\\
&= P_{t|t-1} H^\top R^{-1} (z_t - H \hat x_{t|t-1} - H(\hat x_{t|t} - \hat x_{t|t-1}) )
\end{align}
$$

Therefore,

$$
\hat x_{t|t} - \hat x_{t|t-1} = (I + P_{t|t-1} H^\top R^{-1} H)^{-1} P_{t|t-1} H^\top R^{-1} (z_t - H \hat x_{t|t}) \equiv K (z_t - H \hat x_{t|t})
$$

where $$K = (I + P_{t|t-1} H^\top R^{-1} H)^{-1} P_{t|t-1} H^\top R^{-1}$$ is the _Kalman gain_.

### State covariances

The covariances

$$
(P_{t|t-1})_{ab} \equiv  \langle  (x_t - \hat x_{t|t-1})_a (x_t - \hat x_{t|t-1})_b \rangle
$$

$$
(P_{t|t})_{ab} \equiv \langle (x_t - \hat x_{t|t})_a (x_t - \hat x_{t|t})_b \rangle
$$

where



$$
\begin{align}
(P_{t|t})_{ab} &= \langle [x_t - (\hat x_{t|t} - \hat x_{t|t-1}) - \hat x_{t|t-1}]_a [\cdots]_b \rangle \\
&= \langle [x_t - \hat x_{t|t-1} + (\hat x_{t|t} - \hat x_{t|t-1})]_a [\cdots]_b \rangle \\
&= \langle [x_t - \hat x_{t|t-1} - K (z_t - H\hat x_{t|t-1})]_a [\cdots]_b \rangle \\
&= \langle [x_t - \hat x_{t|t-1} - K (H(x_t-\hat x_{t|t-1}) + \xi)]_a [\cdots]_b \rangle \\
&= \langle [(I-KH)_{ac}(x_t - \hat x_{t|t-1})_c - K_{ac}\xi_c][(I-KH)_{bd}(x_{t} - \hat x_{t|t-1})_d - K_{bd}\xi_d]  \rangle\\
&= (I - KH) P_{t|t-1} (I - KH)^\top + KRK^\top
\end{align}
$$

Given the prior covariance, minimize the posterior MSE $$\min_K (P_{t|t})_{aa}$$

$$
\begin{align}
\frac{\partial\, {\rm tr} P_{t|t}}{\partial K}
&= (-HP_{t|t-1}(I-KH)^\top + RK^\top)^\top \\
&= KR - (I-KH)P_{t|t-1}^\top H^\top \\
&= K (R + HP_{t|t-1}^\top H^\top) - P_{t|t-1}^\top H^\top
\end{align}
$$

Therefore,

$$
K^* = P_{t|t-1}^\top H^\top (HP_{t|t-1}^\top H^\top + R)^{-1}
$$

But previously we got $$K = (I + P_{t|t-1} H^\top R^{-1} H)^{-1} P_{t|t-1} H^\top R^{-1}$$.&#x20;

Are they consistent? To show that

$$
\begin{align}
K^{-1} &= R (P_{t|t-1}H^\top)^{-1}(I + P_{t|t-1} H^\top R^{-1} H) \\
&= R(P_{t|t-1}H^\top)^{-1} + H \\
&= (R + HP_{t|t-1}H^\top)(P_{t|t-1}H^\top)^{-1}
\end{align}
$$

Thus, $$K = (P_{t|t-1}H^\top) (R + HP_{t|t-1}H^\top)^{-1} = K^*$$&#x20;

Given the posterior, the prior covariance for next step

$$
\begin{align}
(P_{t|t-1})_{ab} &\equiv  \langle  (x_t - \hat x_{t|t-1})_a (x_t - \hat x_{t|t-1})_b \rangle \\
&= \langle  (x_t - F\hat x_{t-1|t-1})_a (x_t - F\hat x_{t-1|t-1})_b \rangle \\
&= \langle  (x_t - F x_{t-1|t-1} + F(x_{t-1|t-1} - \hat x_{t-1|t-1}) )_a (x_t - F x_{t-1|t-1} + F(x_{t-1|t-1} - \hat x_{t-1|t-1}))_b \rangle
\end{align}
$$

Thus, the forecasted covariance the composed of measurement noise and evolution noise

$$
P_{t|t-1} = Q + F P_{t-1|t-1} F^\top
$$

### Algorithm

* Forecast state mean using posterior as prior

$$
\hat x_{t|t-1} = F\hat x_{t-1 | t-1}\\ P_{t|t-1} = Q + F P_{t-1|t-1} F^\top
$$

* Compute Kalman gain $$K=(P_{t|t-1}H^\top)(R + HP_{t|t-1}H^\top)^{-1}$$
* Make a measurement $$z_t$$and a prediction using forecast state __ $$H\hat x_{t|t-1}$$__
* Compute posterior mean using new observation $$\hat x_{t|t} = \hat x_{t|t-1} + K(z_t - H\hat x_{t|t-1})$$
* Compute posterior covariance $$P_{t|t} = (I -KH)P_{t|t-1}(I-KH)^\top + K R K^\top$$
