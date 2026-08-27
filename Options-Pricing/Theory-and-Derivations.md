# Derivations and Theory

The following is an accompanying document to the notebook and .py files in this project. It aims to explain key mathematical results clearly and provide technical details for the numerical implementations used. The notation used throughout matches the notation used in the code for the numerical solvers.

## Contents

1. [Black-Scholes Derivation](#1-black-scholes-derivation-and-solution)
    - 1.1 [Derivation](#11-derivation)
    - 1.2 [Transformation to heat equation](#12-transformation-to-heat-equation)
    - 1.3 [Analytical solution](#13-analytical-solution)
2. [Binomial tree method](#2-binomial-tree-method)
3. [Finite difference scheme](#3-finite-difference-scheme)

## 1 Black-Scholes Derivation and Solution

### 1.1 Derivation
[↑ Back to top](#contents)

We will begin with a derivation of the Black-Scholes equation, from now on abbreviated to BS. The key assumptions are:

- Constant volatility, $\sigma$
- Constant risk-free interest rate, $r$
- Stock prices follow a geometric Brownian motion (GBM)
- No dividends are paid out

Under these assumptions, we begin with modelling the stock price as a stochastic process. Since it follows A GBM, it must satisy the stochastic differential equation (SDE)

$$
dS = \mu S + \sigma S dB,
$$

where $\mu$ is some drift parameter. The value of an option, $V$, depends only on time and the stock price, thus $V=V(S,t)$. Applying Itô's lemma to $V$, we find that

$$
dV = \frac{\partial V}{\partial t} dt + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} dt + \frac{\partial V}{\partial S} dS
$$

Now, at the same time as doing this we shall consider a delta-hedged portfolio, $\Pi$, consisting of longing a single option and shorting $\Delta$ shares of the underlying stock. We thus have

$$
\Pi = V - \Delta S.
$$

It follows then that any incremental change in the portfolio's value is given by

$$
\begin{align*}
    d\Pi &= dV - \Delta dS \\
    & = \frac{\partial V}{\partial t} dt + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} dt + \frac{\partial V}{\partial S} dS - \Delta dS
\end{align*}
$$

The entire purpose of delta-hedging is to eliminate any dependency of the portfolio's value on changes in the underlying stock, thus it is necessary to choose $\Delta = \frac{\partial V}{\partial S}$ leaving us with

$$
d\Pi = \frac{\partial V}{\partial t} dt + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} dt
$$

only. Now that we have eliminated any dependency on the underlying stock, the only change in the portfolio's value should be due to interest. We should thus have, for an incremental time step,

$$
d\Pi = r\Pi \, dt = rV \, dt - rS\frac{\partial V}{\partial S} dt.
$$

Equating these two separate expressions for $d\Pi$ yields

$$
\begin{align*}
    &rV \, dt - rS\frac{\partial V}{\partial S} dt = \frac{\partial V}{\partial t} dt + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} dt \\
    \implies &rV = \frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2},
\end{align*}
$$

and we have our desired result, the BS equation.

In the case of a call option with strike price $K$, the expected payoff at maturity is $\max{(S-K, 0)}$, thus we must impose the terminal condition $V(S,T) = \max{(S-K, 0)}$ where $T$ is the maturity date. We also require that $V(0,t)=0$ for all $t$ since a stock of no value cannot grow under a GBM. Similarly, we impose the far field condition $V \sim S-Ke^{-r(T-t)}$ as $S \rightarrow \infty$ since for sufficiently large stock prices, the option will almost certainly be exercised.

In the case of a put option the corresponding conditions are

$$
V(S,t) \sim 0 \text{ as } S\rightarrow \infty, \\
V(0,t) = Ke^{-r(T-t)}, \\
V(S,T) = \max{(K-S, 0)}.
$$

### 1.2 Transformation to heat equation
[↑ Back to top](#contents)

The BS equation does, in fact, have an analytical solution. To see this, we use the substitution

$$
\tau = \frac{\sigma^2}{2}(T-t), \quad x = \ln\bigg(\frac{S}{K}\bigg) + \bigg(r-\frac{\sigma^2}{2}\bigg)(T-t), \quad u(x, \tau) = Ve^{2r\tau/\sigma^2}.
$$

By the chain rule, we have

$$
\begin{align*}
    \frac{\partial}{\partial t} &= \frac{\partial x}{\partial t}\frac{\partial}{\partial x} + \frac{\partial \tau}{\partial t}\frac{\partial}{\partial \tau} = \bigg(\frac{\sigma^2}{2} - r\bigg)\frac{\partial}{\partial x} - \frac{\sigma^2}{2}\frac{\partial}{\partial \tau}, \\
    \frac{\partial}{\partial S} &= \frac{\partial x}{\partial S}\frac{\partial}{\partial x} + \frac{\partial \tau}{\partial S}\frac{\partial}{\partial \tau} = \frac{1}{S}\frac{\partial}{\partial x}.
\end{align*}
$$

We may thus rewrite the BS equation as

$$
\begin{align*}
    rue^{-2r\tau/\sigma^2} &= \bigg(\frac{\sigma^2}{2} - r\bigg)\frac{\partial (ue^{-2r\tau/\sigma^2})}{\partial x} - \frac{\sigma^2}{2}\frac{\partial (ue^{-2r\tau/\sigma^2})}{\partial \tau} + r \frac{\partial (ue^{-2r\tau/\sigma^2})}{\partial x} + \frac{\sigma^2 S}{2}\frac{\partial}{\partial x}\bigg( \frac{1}{S}\frac{\partial (ue^{-2r\tau /\sigma^2})}{\partial x}\bigg) \\
    rue^{-2r\tau/\sigma^2} &= e^{-2r\tau/\sigma^2}\bigg(\frac{\sigma^2}{2} - r\bigg)\frac{\partial u}{\partial x} - e^{-2r\tau/\sigma^2}\frac{\sigma^2}{2}\frac{\partial u}{\partial \tau} + rue^{-2r\tau/\sigma^2} + re^{-2r\tau/\sigma^2} \frac{\partial u}{\partial x} - e^{-2r\tau/\sigma^2}\frac{\sigma^2}{2}\frac{\partial u}{\partial x} + e^{-2r\tau/\sigma^2}\frac{\sigma^2}{2}\frac{\partial^2u}{\partial x^2} \\
    \frac{\partial u}{\partial \tau}  &=  \frac{\partial^2u}{\partial x^2},
\end{align*}
$$

which we recognise as the well-known heat equation. The corresponding transformed boundary conditions are

$$
\lim_{x\rightarrow -\infty} u(x,\tau) = 0, \\
u(x, \tau) \sim K(e^{x+\tau}-1) \text{ as } x\rightarrow \infty, \\
u(x,0) = K(e^x-1)H(x),
$$

where $H(x)$ is the Heaviside step function.

For a put option, the corresponding conditions are

$$
\lim_{x\rightarrow \infty} u(x, \tau) = 0, \\
u(x,\tau) \sim K(1-e^{x+\tau}) \text{ as } x \rightarrow -\infty, \\
u(x, 0) = K(1-e^x)H(-x).
$$

### 1.3 Analytical solution
[↑ Back to top](#contents)

Since this heat equation is defined on an unbounded spatial domain, we use the standard method of taking the convolution of the initial condition with the fundamental solution of the heat equation. In the 1D case, as applies here, we have

$$
\begin{align*}
    u(x, \tau) &= \frac{1}{(4\pi\tau)^{\frac{1}{2}}} \int_{-\infty}^\infty K(e^y-1)H(y)e^{-\frac{(x-y)^2}{4\tau}} dy \\
    &= \frac{K}{(4\pi\tau)^{\frac{1}{2}}} \int_0^\infty (e^y-1) e^{-\frac{(x-y)^2}{4\tau}} dy \\
    &= \frac{K}{(4\pi\tau)^{\frac{1}{2}}} \bigg( I_1 - I_2\bigg)
\end{align*}
$$

where $I_1=\int_0^\infty e^{y-\frac{(x-y)^2}{4\tau}} dy$ and $I_2 = \int_0^\infty e^{-\frac{(x-y)^2}{4\tau}} dy$. We compute these integrals in turn. By completing the square in the exponent we have

$$
\begin{align*}
    I_1 &= \int_0^\infty \exp\bigg(\frac{4\tau y - x^2 - y^2 +2xy}{4\tau}\bigg) dy \\
    &= e^{\tau + x}\int_0^\infty \exp\bigg(-\frac{(y-(2\tau+x))^2}{4\tau}\bigg) dy.
\end{align*}
$$

The substitution $z=\frac{y-(2\tau+x)}{\sqrt{2\tau}}$ yields

$$
\begin{align*}
    I_1 &= \sqrt{2\tau} e^{\tau + x} \int_{-\frac{2\tau + x}{\sqrt{2\tau}}}^\infty e^{-\frac{z^2}{2}} dz \\
    &= \sqrt{4\pi\tau} e^{\tau + x} \Phi\bigg(\frac{2\tau + x}{\sqrt{2\tau}}\bigg),
\end{align*}
$$

where $\Phi$ is the CDF of a standard normal random variable and the last line follows from the symmetry of a standard normal distribution about $0$. The remaining integral is computed using the substitution $z=\frac{x-y}{\sqrt{2\tau}}$ which yields

$$
I_2 = -\sqrt{2\tau}\int_{\frac{x}{\sqrt{2\tau}}}^{-\infty} e^{-\frac{z^2}{2}} dz = \sqrt{4\pi\tau}\Phi\bigg(\frac{x}{\sqrt{2\pi}}\bigg).
$$

Again, the last equality follows from the symmetrical properties of a standard normal PDF. Combining these results we see that

$$
u(x,\tau) = K\Bigg( e^{\tau+x}\Phi\bigg(\frac{2\tau + x}{\sqrt{2\tau}}\bigg) - \Phi\bigg(\frac{x}{\sqrt{2\pi}}\bigg)\Bigg).
$$

In terms of the original variables, this is

$$
V(S,t) = S\Phi\bigg(\frac{\ln(\frac{S}{K})+(r+\frac{\sigma^2}{2})(T-t)}{\sigma \sqrt{T-t}} \bigg) - Ke^{-r(T-t)}\Phi\bigg(\frac{\ln(\frac{S}{K}) + (r-\frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}} \bigg),
$$

which is our desired analytic solution to the BS equation.

In the case of a put option, the analysis is very similar. We note that the only difference is in the boundary and initial conditions. We find instead that

$$
u(x,\tau) = \frac{K}{(4\pi\tau)^{\frac{1}{2}}} \bigg( J_1 - J_2\bigg),
$$

where

$$
\begin{align*}
    J_1 &= \int_{-\infty}^0 e^{-\frac{(x-y)^2}{4\tau}} dy = \sqrt{4\tau \pi} \Phi\bigg( - \frac{x}{\sqrt{2\tau}} \bigg), \\
    J_2 &= \int_{-\infty}^0 e^{y-\frac{(x-y)^2}{4\tau}} dy = \sqrt{4\tau \pi} e^{\tau+x}\Phi\bigg( - \frac{2\tau + x}{\sqrt{2\tau}} \bigg).
\end{align*}
$$

It thus follows that the value of a put option is given by

$$
V(S,t) = Ke^{-r(T-t)}\Phi\bigg(-\frac{\ln(\frac{S}{K}) + (r-\frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}} \bigg) - S\Phi\bigg(-\frac{\ln(\frac{S}{K})+(r+\frac{\sigma^2}{2})(T-t)}{\sigma \sqrt{T-t}} \bigg).
$$

## 2 Binomial tree method
[↑ Back to top](#contents)

## 3 Finite difference scheme
[↑ Back to top](#contents)
