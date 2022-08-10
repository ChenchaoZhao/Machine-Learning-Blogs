# Finite Step Markov Diffusion Models

Let $p( \cdot | \theta )$ be the parametric model that models data $x\_0 \sim q\_0$, then we can optimize $\theta$ by maximize the likelihood $p(x\_0 | \theta)$ or equivalently

$$
\max_\theta \mathbb E_{x_0 \sim q_0} \log p(x_0|\theta).
$$

From now on, we use $q$'s to denote the forward physical distributions and $p$'s the backward variational ansatz. The parameters are implied in $p(\cdot|\theta) \equiv p(\cdot)$.

Let $T > 1$ be the diffusion steps, the joint density of forward process $q(x\_{0, 1, \cdots T})$ can be expanded sequentially if the process is Markov

$$
q(x_0, x_1, \cdots, x_T) = q_0(x_0) \prod_{t=1}^T q_{t|t-1}(x_t|x_{t-1}).
$$

The reverse process variational ansatz can be similarly constructed

$$
p(x_0,x_1, \cdots, x_T) = p_T(x_T) \prod_{t=1}^T p_{t-1|t}(x_{t-1}|x_{t})
$$

which can be interpreted as a series of consecutive priors for the physical observation $x\_0$.

If we marginalize the latent variables $x\_{1,\cdots,T}$, we get the objective function or observable likelihood

$$
p(x_0) = \int \mathcal D x_{1,\cdots,T} \; p(x_0,x_1, \cdots, x_T)\equiv\int_{1,\cdots,T} \, p_{0,1,\cdots,T}.
$$

### MLE and variational approach

We will derive the lower bound of maximum likelihood objective $\mathbb E\_{x\_0 \sim q\_0} \log p(x\_0)$ and then show that it is related to a KL-divergence up to a constant.

_Proof._ The original approach would expand the $p$ joint density

$$
p(x_0) = \int_{1,\cdots,T} \; p_T(x_T) \prod_{t=1}^T p_{t-1|t}(x_{t-1}|x_{t})
$$

assume Markov property of backward process. However, without assumption of Markov property, we can still insert an identity and get

$$
p(x_0) = \int_{1,\cdots,T} \; \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}}q_{0,1,\cdots,T}.
$$

Note that $q\_{0,1,\cdots,T} = q\_{1,\cdots,T | 0} q\_0$, then we can reinterpret the integral

$$
p(x_0) = q_0\mathbb E_{q_{1,\cdots,T|0}} \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}}.
$$

The log-likelihood average over all data distribution is

$$
\mathbb E_{q_0} \log p(x_0) = \mathbb E_{q_0} \log q_0 +\mathbb E_{q_0} \log \mathbb E_{q_{1,\cdots,T|0}} \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}}.
$$

Use the concavity of logarithm function,

$$
\mathbb E_{q_0} \log p(x_0) \ge \mathbb E_{q_0} \log q_0 + \mathbb E_{q_{0,1,\cdots,T}} \log \frac{p_{0,1,\cdots,T}}{q_{0,1,\cdots,T}} = L_{\rm ELBO}.
$$

where

$$
L_{\rm ELBO} = - H[q_0] - D_{\rm KL} (q_{0,\cdots,T}|p_{0,\cdots,T}).
$$

The entropy $H\[q\_0]$ depends on data distribution and does not contain model parameters. Thus, maximizing log-likelihood lower bound $L\_{\rm ELBO}$ is equivalent to minimizing KL-divergence between forward process joint density and backward process joint density.

> Regardless of Markov property of the processes, max likelihood lower bound is equivalent to min KL-divergence between variational ansatz and physical forward process.

### Markov variational Ansatz

If the forward process is Markov, then we have

$$
q_{0,1,\cdots,T} = q_0 \prod_{t=1}^T q_{t|t-1}.
$$

Similarly, if we assume the backward joint density can be expanded as

$$
p_{0,1,\cdots,T} = \left(\prod_{t=1}^T p_{t-1|t} \right) p_T.
$$

The posterior of physical forward process may be represented as

$$
q_{t-1|t} \propto q_{t|t-1} q_{t-1};
$$

however, without the knowledge of the initial state $x\_0$, there could be infinity possibilities. Therefore, we fix the initial state, and get probabilies given the fixed $x\_0 \sim q\_0$

$$
q_{t-1|t, 0} \propto q_{t|t-1, 0} q_{t-1 | 0},
$$

or the equality for $t>1$

$$
q_{t-1|t, 0} q_{t|0} = q_{t, t-1|0} = q_{t|t-1, 0} q_{t-1 | 0}.
$$

Thus, the forward process joint density has an posterior expansion

$$
q_{0,1,\cdots,T} = q_0 q_{1|0}\prod_{t=2}^T q_{t|t-1,0} = q_0 q_{1|0}\prod_{t=2}^T q_{t-1|t,0}\frac{q_{t|0}}{q_{t-1|0}}
$$

where the last factor telescopes

$$
q_{0,1,\cdots,T} = q_0 \left( \prod_{t=2}^T q_{t-1|t,0} \right) q_{T|0}.
$$

The ratio of forward and backward density can be expanded in the following fashion

$$
\frac{q_{0,1,\cdots,T}}{p_{0,1,\cdots,T}} =\frac{q_{T|0}\,q_0}{p_T\,p_{0|1}}\prod_{t=2}^T\frac{q_{t-1|t,0}}{p_{t-1|t,0}}
$$

whose logarithm reads

$$
\log\frac{q_{0,1,\cdots,T}}{p_{0,1,\cdots,T}} =\log\frac{q_0}{p_{0|1}} + \sum_{t=2}^T\log\frac{q_{t-1|t,0}}{p_{t-1|t,0}} +\log\frac{q_{T|0}}{p_T} .
$$

Using the posterior expansion of $$q$$, the total KL-divergence

$$
D_{\rm KL} (q_{0,\cdots,T}|p_{0,\cdots,T}) \equiv \sum_{t=1}^T D_{t-1}
$$

where

* $$D_0 = D_{\rm KL}(q_0|p_{0|1})$$,
* $$D_{t-1} = D_{\rm KL}(q_{t-1|t,0}|p_{t-1|t,0})$$ for $$1<t<T$$, and
* $$D_T = D_{\rm KL} (q_{T|0}|p_T)$$.

The last term $$D_T$$  is a constant with **fixed** distribution $$p_T$$. If add back the entropy term $$H[q_0]$$, the first term becomes the usual likelihood $$L_0 = D_0 + H[q_0] = -\log p_{0|1}$$and the total loss becomes a typical variational inference loss: the sum of data negative log-likelihood and a series of prior KL-divergence.

> So far we have not assumed any specific distribution yet. The objective is purely based on the assumption of Markov property.

#### Gaussian diffusion

For the particular case of Gaussian diffusion models, we assume

* the terminal distribution is normal

$$
q_{T|0} = q_T=p_T = \mathcal N({\bf x}_T;{\bf 0}, {\bf 1}),
$$

* forward transition process is Gaussian

$$$
q_{t|t-1} = \mathcal N({\bf x}_t; \sqrt{1-\beta_t}{\bf x}_{t-1}, \beta_t {\bf 1})$$ where $0<\beta_t \le 1$. It is useful to introduce additional notations: * $\alpha_t \equiv 1 - \beta_t$, and * $\bar \alpha_t \equiv \prod_{t=1}^T \alpha_t$. #### Reparameterization Let ${\bf z}\sim \mathcal N({\bf 0}, {\bf 1})$, the forward process can be written as
$$$

{\bf x}_t = \sqrt{1-\beta\_t} {\bf x}_{t-1} + \sqrt{\beta\_t} {\bf z} =\sqrt{\alpha\_t} {\bf x}\_{t-1} + \sqrt{\beta\_t} {\bf z} .

$$
Iteratively apply the formula, we get
$$

{\bf x}\_t = \sqrt{\bar \alpha\_t} {\bf x}\_0 + \sqrt{1-\bar \alpha\_t} {\bf z}.

$$
> Thus, we can generate ${\bf x}_t$ for **any** $t$ without actually do the iterative calculations. > There is a similar property for any Markov process, i.e. Feynman-Kac formula #### Posterior is Gaussian We can compute the posterior of the physical process using
$$

q\_{t-1|t, 0} \propto q\_{t|t-1, 0} q\_{t-1 | 0}

$$
where the RHS is a product of Gaussians. One can show that the posterior is indeed Gaussian after doing an easy but lengthy calculation
$$

q\_{t-1|t, 0} = \mathcal N({\bf x}\_{t-1}; \tilde \mu\_t({\bf x}\_t, {\bf x}\_0), \tilde \beta\_t {\bf 1})

$$
where
$$

\tilde \mu\_t = \frac{\sqrt\alpha\_t (1 - \bar \alpha\_{t-1}){\bf x}_t + \beta\_t \sqrt{\bar \alpha_{t-1\}}{\bf x}\_0}{1-\bar\alpha\_t} ,

$$
and
$$

\tilde \beta\_t = \beta\_t \frac{1-\bar\alpha\_{t-1\}}{1-\alpha\_t} .

$$
Express ${\bf x}_0$ in terms of ${\bf x}_t$ and noise, the mean simplies
$$

\tilde \mu\_t ({\bf x}\_t) = \alpha\_t^{-\frac12 } \left( {\bf x}\_t - \frac{\beta\_t}{\sqrt{1-\bar \alpha\_t\}} {\bf z} \right).

$$
#### Variational Ansatz Since the target distribution is Gaussian, it is a good idea to choose Gaussian distribution as the variational Ansatz
$$

p\_{t-1|t} = \mathcal N ({\bf x}\_{t-1}; \mu\_t, \sigma^2\_t{\bf 1} )

$$
where the model parameters are $\mu_t$ and $\sigma_t$. The variance will eventually contribute to learning rate; we will treat $\sigma_t$ as a hyperparameter instead of learning it from stochastic gradient descent. The only learnable parameter is then $\mu_t=\mu_t({\bf x}_t, t)$. Recall the objective for each time step $1<t<T$
$$

D\_{t-1} = D\_{\rm KL}(q\_{t-1|t,0}|p\_{t-1|t,0}) = \frac{1}{2\sigma^2\_t} \Vert \mu\_t - \tilde \mu\_t \Vert^2 + {\rm const.}

$$
where we used the KL-divergence between two Gaussian distributions. Clearly, we have the exact solution
$$

\frac{\delta D\_{t-1}\[\mu\_t]}{\delta \mu\_t} = 0 \Rightarrow \mu\_t = \tilde \mu\_t,

$$
and therefore,
$$

\mu\_t ({\bf x}\_t, t) = \alpha\_t^{-\frac12 } \left( {\bf x}\_t - \frac{\beta\_t}{\sqrt{1-\bar \alpha\_t\}} {\bf z}\_t \right)

$$
where ${\bf z}_t\sim\mathcal N(0, {\bf 1})$ is the noise that generated ${\bf x}_t$ from ${\bf x}_0$. Next, we reparameterize $\mu_t$ to separate the *explicit* dependency of ${\bf x}_t$ and $t$ and let the model only focus on the *implicit* dependencies, i.e.
$$

\mu\_t ({\bf x}\_t, t) = \alpha\_t^{-\frac12 } \left( {\bf x}\_t - \frac{\beta\_t}{\sqrt{1-\bar \alpha\_t\}} {\bf z}({\bf x}\_t, t) \right).

$$
Finally, the objective becomes
$$

D\_{t-1} = \frac{\beta\_t^2}{2\alpha\_t(1-\bar\alpha\_t)\sigma\_t^2} \Vert {\bf z}\_t -{\bf z}({\bf x}\_t,t)\Vert^2

$$
where ${\bf z}({\bf x}_t, t)$ is the model output. It can also be shown that
$$

L\_0 = \frac{\beta\_1}{2\alpha\_1\sigma\_t^2} \Vert {\bf z}\_1 -{\bf z}({\bf x}\_0,1)\Vert^2

$$
where $\bar \alpha_1 = \alpha_1 = 1 -\beta_1$. Thus, the generic loss term for $0 < t < T$ is
$$

L\_{t-1} = \frac{\beta\_t^2}{2\alpha\_t(1-\bar\alpha\_t)\sigma\_t^2} \Vert {\bf z}\_t -{\bf z}({\bf x}\_t,t)\Vert^2.

$$
Note that we still have the freedom to choose $\sigma_t$ that controls the importance of each step. But in the literature, they usually take a *heuristic* approach by ignoring the weight factor keeping only the $\ell_2$ loss. #### Sampling the backward process During training, the model learned the backward transition distribution
$$

p\_{t-1|t} = \mathcal N \left ({\bf x}\_{t-1}; \alpha\_t^{-\frac12 } \left( {\bf x}\_t - \frac{\beta\_t}{\sqrt{1-\bar \alpha\_t\}} {\bf z}({\bf x}\_t, t) \right), \sigma^2\_t{\bf 1} \right).

$$
The backward iteration is essentially sampling and calculating
$$

{\bf x}\_{t-1} = \alpha\_t^{-\frac12 } \left( {\bf x}\_t - \frac{\beta\_t}{\sqrt{1-\bar \alpha\_t\}} {\bf z}({\bf x}\_t, t) \right) + \sigma\_t {\bf z}

$$
where ${\bf z} \sim \mathcal N({\bf 0}, {\bf 1})$. #### Training and inference algorithms ##### Training * Sample * ${\bf x}_0 \sim q_0$ * $t \sim {\sf Uniform}(1,\cdots,T)$ * ${\bf z}_t \sim \mathcal N({\bf 0}, {\bf 1})$ * Construct ${\bf x}_t$ * Feed ${\bf x}_t, t$ to model * Minimize $L_{t-1}$ ##### Inference * Sample $x_T \sim \mathcal N({\bf 0}, {\bf 1})$ * Loop $t = T,\cdots, 1$ * Sample ${\bf z} \sim \mathcal N({\bf 0}, {\bf 1})$ * Compute ${\bf x}_{t-1}$ * Return ${\bf x}_0$ The reconstruction formula is given in previous section " sampling the backward process." <!--stackedit_data: eyJoaXN0b3J5IjpbNTEyMTQ3NDAyLDEwMjgzOTQyMiwtMTkyNT AwMjAzNiwtMjAzNDgwNjk3MywxNjg4MTMyNTU5LDExNzAwMDgx MiwtMjk5NDEwMTQ5LC0xODYyMTk3MzczLDE5NDM0ODg4MjEsMT E4ODgyNDkzMywtMTg4NjkwNjI5MSwtMTMyNjcyMzExMCwtNjEy MTE3MTM3LDExNDc5NTQ4ODAsLTk4NTU0NTkxMywtMTc0NjY3MT k1NiwtMTQ1NjI0MDA5LC0xMzk0OTkwMTc5LC00MjcwNTMxMjYs MTMyOTgxMjU0M119 -->
$$
