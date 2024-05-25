```@meta
CurrentModule = IncompressibleNavierStokes
```

# Time discretization

The spatially discretized Navier-Stokes equations form a differential-algebraic
system, with an ODE for the velocity

```math
\frac{\mathrm{d} u}{\mathrm{d} t} = F(u, t) - (G p + y_G)
```

subject to the algebraic constraint formed by the mass equation

```math
M u + y_M = 0.
```

In the end of the previous section, we differentiated the mass
equation in time to obtain a discrete pressure Poisson equation. This equation
includes the term ``\frac{\mathrm{d} y_M}{\mathrm{d} t}``, which is non-zero if
an unsteady flow of mass is added to the domain (Dirichlet boundary
conditions). This term ensures that the time-continuous discrete velocity field
``u(t)`` stays divergence free (conserves mass). However, if we directly
discretize this system in time, the mass preservation may actually not be
respected. For this, we will change the definition of the pressure such that
the time-discretized velocity field is divergence free at each time step and
each time sub-step (to be defined in the following).

Consider the interval ``[0, T]`` for some simulation time ``T``. We will divide
it into ``N`` sub-intervals ``[t^n, t^{n + 1}]`` for ``n = 0, \dots, N - 1``,
with ``t^0 = 0``, ``t^N = T``, and increment ``\Delta t^n = t^{n + 1} - t^n``.
We define ``u^n \approx u(t^n)`` as an approximation to the exact discrete
velocity field ``u(t^n)``, with ``u^0 = u(0)`` starting from the exact
initial conditions. We say that the time integration scheme (definition of
``u^n``) is accurate to the order ``r`` if ``u^n = u(t^n) +
\mathcal{O}(\Delta t^r)`` for all ``n``.

IncompressibleNavierStokes provides a collection of explicit and implicit
Runge-Kutta methods, in addition to Adams-Bashforth Crank-Nicolson and one-leg
beta method time steppers.

The code is currently not adapted to time steppers from
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/),
but they may be integrated in the future.

```@docs
AbstractODEMethod
AbstractRungeKuttaMethod
isexplicit
lambda_conv_max
lambda_diff_max
ode_method_cache
runge_kutta_method
timestep
timestep!
```

## Explicit Runge-Kutta methods

See Sanderse [Sanderse2012](@cite).

Consider the velocity field ``u_0`` at a certain time ``t_0``. We will now
perform one time step to ``t = t_0 + \Delta t``. For explicit Runge-Kutta
methods, this time step is divided into ``s`` sub-steps ``t_i = t_0 + \Delta
t_i`` with increment ``\Delta t_i = c_i \Delta t``. The final substep performs
the full time step ``\Delta t_s = \Delta t`` such that ``t_s = t``.

For ``i = 1, \dots, s``, the intermediate velocity ``u_i`` and pressure ``p_i``
are computed as follows:

```math
\begin{split}
k_i & = F(u_{i - 1}, t_{i - 1}) - y_G(t_{i - 1}) \\
v_i & = u_0 + \Delta t \sum_{j = 1}^i a_{i j} k_j \\
L p_i & = W M \frac{1}{c_i} \sum_{j = 1}^i a_{i j} k_j +
W \frac{y_M(t_i) - y_M(t_0)}{\Delta t_i} \\
& = W \frac{(M v_i + y_M(t_i)) - (M u_0 + y_M(t_0))}{\Delta t_i^n} \\
& = W \frac{M v_i + y_M(t_i)}{\Delta t_i^n} \\
u_i & = v_i - \Delta t_i G p_i,
\end{split}
```

where ``(a_{i j})_{i j}`` are the Butcher tableau coefficients of the
RK-method, with the convention ``c_i = \sum_{j = 1}^i a_{i j}``.

Finally, we return ``u_s``. If ``u_0 = u(t_0)``, we get the accuracy ``u_s =
u(t) + \mathcal{O}(\Delta t^{r + 1})``, where ``r`` is the order of the
RK-method. If we perform ``n`` RK time steps instead of one, starting at exact
initial conditions ``u^0 = u(0)``, then ``u^n = u(t^n) + \mathcal{O}(\Delta
t^r)`` for all ``n \in \{1, \dots, N\}``. Note that for a given ``u``, the
corresponding pressure ``p`` can be calculated to the same accuracy as ``u`` by
doing an additional pressure projection after each outer time step ``\Delta t``
(if we know ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t)``), or to first order
accuracy by simply returning ``p_s``.

Note that each of the sub-step velocities ``u_i`` is divergence free, after
projecting the tentative velocities ``v_i``. This is ensured due to the
judiciously chosen replacement of ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t_i)``
with ``(y_M(t_i) - y_M(t_0)) / \Delta t_i``. The space-discrete
divergence-freeness is thus perfectly preserved, even though the time
discretization introduces other errors.

```@docs
ExplicitRungeKuttaMethod
```

## Implicit Runge-Kutta methods

See Sanderse [Sanderse2013](@cite).

```@docs
ImplicitRungeKuttaMethod
```

## Adams-Bashforth Crank-Nicolson method

We here require that the time step ``\Delta t`` is constant. This methods uses
Adams-Bashforth for the convective terms and Crank-Nicolson stepping for the
diffusion and body force terms. Given the velocity field ``u_0 = u(t_0)`` at
a time ``t_0`` and its previous value ``u_{-1} = u(t_0 - \Delta t)`` at the
previous time ``t_{-1} = t_0 - \Delta t``, the predicted velocity field ``u``
at the time ``t = t_0 + \Delta t`` is defined by first computing a tentative
velocity:

```math
\begin{split}
\frac{v - u_0}{\Delta t}
& = - (\alpha_0 C(u_0, t_0) + \alpha_{-1} C(u_{-1}, t_{-1})) \\
& + \theta (D u_0 + y_D(t_0)) + (1 - \theta) (D v + y_D(t)) \\
& + \theta f(t_0) + (1 - \theta) f(t) \\
& - (G p_0 + y_G(t_0)),
\end{split}
```

where ``\theta \in [0, 1]`` is the Crank-Nicolson parameter (``\theta =
\frac{1}{2}`` for second order convergence), ``(\alpha_0, \alpha_{-1}) = \left(
\frac{3}{2}, -\frac{1}{2} \right)`` are the Adams-Bashforth coefficients, and
``v`` is a tentative velocity yet to be made divergence free. We can group the
terms containing ``v`` on the left hand side, to obtain

```math
\begin{split}
\left( \frac{1}{\Delta t} I - (1 - \theta) D \right) v
& = \left(\frac{1}{\Delta t} I - \theta D \right) u_0 \\
& - (\alpha_0 C(u_0, t_0) + \alpha_{-1} C(u_{-1}, t_{-1})) \\
& + \theta y_D(t_0) + (1 - \theta) y_D(t) \\
& + \theta f(t_0) + (1 - \theta) f(t) \\
& - (G p_0 + y_G(t_0)).
\end{split}
```

We can compute ``v`` by inverting the positive definite matrix ``\left(
\frac{1}{\Delta t} I - \theta D \right)`` for the given right hand side using a
suitable linear solver. Assuming ``\Delta t`` is constant, we can precompute a
Cholesky factorization of this matrix before starting time stepping.

We then compute the pressure difference ``\Delta p`` by solving

```math
L \Delta p = W \frac{M v + y_M(t)}{\Delta t} - W M (y_G(t) - y_G(t_0)),
```

after which a divergence free velocity ``u`` can be enforced:

```math
u = v - \Delta t (G \Delta p + y_G(t) - y_G(t_0)).
```

A first order accurate prediction of the corresponding pressure is ``p = p_0 +
\Delta p``. However, since this pressure is reused in the next time step, we
perform an additional pressure solve to avoid accumulating first order errors.
The resulting pressure ``p`` is then accurate to the same order as ``u``.

```@docs
AdamsBashforthCrankNicolsonMethod
```

## One-leg beta method

See Verstappen and Veldman [Verstappen2003](@cite) [Verstappen1997](@cite).

We here require that the time step ``\Delta t`` is constant. Given the velocity
``u_0`` and pressure ``p_0`` at the current time ``t_0`` and their previous
values ``u_{-1}`` and ``p_{-1}`` at the time ``t_{-1} = t_0 - \Delta t``, we
start by computing the "offstep" values ``v = (1 + \beta) v_0 - \beta v_{-1}``
and ``Q = (1 + \beta) p_0 - \beta p_{-1}`` for some ``\beta = \frac{1}{2}``.

A tentative velocity field ``\tilde{v}`` is then computed as follows:

```math
\tilde{v} = \frac{1}{\beta + \frac{1}{2}} \left( 2 \beta u_0 - \left( \beta -
\frac{1}{2} \right) u_{-1} + \Delta t F(v, t) - \Delta t
(G Q + y_G(t)) \right).
```

A pressure correction ``\Delta p `` is obtained by solving the Poisson equation
```math
L \Delta p = \frac{\beta + \frac{1}{2}}{\Delta t} W (M \tilde{v} + y_M(t)).
```

Finally, the divergence free velocity field is given by

```math
u = \tilde{v} - \frac{\Delta t}{\beta + \frac{1}{2}} G \Delta p,
```

while the second order accurate pressure is given by

```math
p = 2 p_0 - p_{-1} + \frac{4}{3} \Delta p.
```

```@docs
OneLegMethod
```
