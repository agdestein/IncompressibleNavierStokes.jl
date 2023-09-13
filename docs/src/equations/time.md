# Time discretization

The spatially discretized Navier-Stokes equations form a differential-algebraic
system, with an ODE for the velocity

```math
\Omega_h \frac{\mathrm{d} u_h}{\mathrm{d} t} = F(u_h, t) - (G p_h + y_G)
```

subject to the algebraic constraint formed by the mass equation

```math
M u_h + y_M = 0.
```

In the end of the [previous section](spatial.md), we differentiated the mass
equation in time to obtain a discrete pressure Poisson equation. This equation
includes the term ``\frac{\mathrm{d} y_M}{\mathrm{d} t}``, which is non-zero if
an unsteady flow of mass is added to the domain (Dirichlet boundary
conditions). This term ensures that the time-continuous discrete velocity field
``u_h(t)`` stays divergence free (conserves mass). However, if we directly
discretize this system in time, the mass preservation may actually not be
respected. For this, we will change the definition of the pressure such that
the time-discretized velocity field is divergence free at each time step and
each time sub-step (to be defined in the following).

Consider the interval ``[0, T]`` for some simulation time ``T``. We will divide
it into ``N`` sub-intervals ``[t^n, t^{n + 1}]`` for ``n = 0, \dots, N - 1``,
with ``t^0 = 0``, ``t^N = T``, and increment ``\Delta t^n = t^{n + 1} - t^n``.
We define ``U^n \approx u_h(t^n)`` as an approximation to the exact discrete
velocity field ``u_h(t^n)``, with ``U^0 = u_h(0)`` starting from the exact
initial conditions. We say that the time integration scheme (definition of
``U^n``) is accurate to the order ``r`` if ``U^n = u_h(t^n) +
\mathcal{O}(\Delta t^r)`` for all ``n``.


## Explicit Runge-Kutta methods

See Sanderse [Sanderse2012](@cite).

Consider the velocity field ``U_0`` at a certain time ``t_0``. We will now
perform one time step to ``t = t_0 + \Delta t``. For explicit Runge-Kutta
methods, this time step is divided into ``s`` sub-steps ``t_i = t_0 + \Delta
t_i`` with increment ``\Delta t_i = c_i \Delta t``. The final substep performs
the full time step ``\Delta t_s = \Delta t`` such that ``t_s = t``.

For ``i = 1, \dots, s``, the intermediate velocity ``U_i`` and pressure ``P_i``
are computed as follows:

```math
\begin{split}
F_i & = \Omega_h^{-1} F(U_{i - 1}, t_{i - 1}) \\
V_i & = U_0 + \Delta t \sum_{j = 1}^i a_{i j} F_j \\
L P_i & = \frac{1}{c_i} \sum_{j = 1}^i a_{i j} F_j +
\frac{y_M(t_i) - y_M(t_0)}{\Delta t_i} \\
& = \frac{(M V_i + y_M(t_i)) - (M U_0 + y_M(t_0))}{\Delta t_i^n} \\
& = \frac{M V_i + y_M(t_i)}{\Delta t_i^n} \\
U_i & = V_i - \Delta t_i \Omega_h^{-1} (G P_i + y_G(t_i)),
\end{split}
```

where ``(a_{i j})_{i j}`` are the Butcher tableau coefficients of the RK-method,
with the convention ``c_i = \sum_{j = 1}^i a_{i j}``.

Finally, we set ``U = U_s``. If ``U_0 = u_h(t_0)``, we get the accuracy ``U =
u_h(t) + \mathcal{O}(\Delta t^{r + 1})``, where ``r`` is the order of the
RK-method. If we perform ``n`` RK time steps instead of one, starting at exact
initial conditions ``U^0 = u_h(0)``, then ``U^n = u_h(t^n) +
\mathcal{O}(\Delta t^r)`` for all ``n \in \{1, \dots, N\}``. Note that for a
given ``U``, the corresponding pressure ``P`` can be calculated to the same
accuracy as ``U`` by doing an additional pressure projection after each outer
time step ``\Delta t`` (if we know ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t)``),
or to first order accuracy by simply taking ``P = P_s``.

Note that each of the sub-step velocities ``U_i`` is divergence free, after
projecting the tentative velocities ``V_i``. This is ensured due to the
judiciously chosen replacement of ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t_i)``
with ``(y_M(t_i) - y_M(t_0)) / \Delta t_i``. The space-discrete
divergence-freeness is thus perfectly preserved, even though the time
discretization introduces other errors.


## Implicit Runge-Kutta methods

See Sanderse [Sanderse2013](@cite).


## Adams-Bashforth Crank-Nicolson method

We here require that the time step ``\Delta t`` is constant. This methods uses
Adams-Bashforth for the convective terms and Crank-Nicolson stepping for the
diffusion and body force terms. Given the velocity field ``U_0 = u_h(t_0)`` at
a time ``t_0`` and its previous value ``U_{-1} = u_h(t_0 - \Delta t)`` at the
previous time ``t_{-1} = t_0 - \Delta t``, the predicted velocity field ``U``
at the time ``t = t_0 + \Delta t`` is defined by first computing a tentative
velocity:

```math
\begin{split}
\frac{V - U_0}{\Delta t}
& = - (\alpha_0 C(U_0, t_0) + \alpha_{-1} C(U_{-1}, t_{-1})) \\
& + \theta (D U_0 + y_D(t_0)) + (1 - \theta) (D V + y_D(t)) \\
& + \theta f(t_0) + (1 - \theta) f(t) \\
& - (G p_0 + y_G(t_0)),
\end{split}
```

where ``\theta \in [0, 1]`` is the Crank-Nicolson parameter (``\theta =
\frac{1}{2}`` for second order convergence), ``(\alpha_0, \alpha_{-1}) = \left(
\frac{3}{2}, -\frac{1}{2} \right)`` are the Adams-Bashforth coefficients, and
``V`` is a tentative velocity yet to be made divergence free. We can group the
terms containing ``V`` on the left hand side, to obtain

```math
\begin{split}
\left( \frac{1}{\Delta t} I - \theta D \right) V
& = \left(\frac{1}{\Delta t} I - (1 - \theta) D \right) U_0 \\
& - (\alpha_0 C(U_0, t_0) + \alpha_{-1} C(U_{-1}, t_{-1})) \\
& + \theta y_D(t_0) + (1 - \theta) y_D(t) \\
& + \theta f(t_0) + (1 - \theta) f(t) \\
& - (G P_0 + y_G(t_0)).
\end{split}
```

We can compute ``V`` by inverting the positive definite matrix ``\left(
\frac{1}{\Delta t} I - \theta D \right)`` for the given right hand side using a
suitable linear solver. Assuming ``\Delta t`` is constant, we can precompute a
Cholesky factorization of this matrix before starting time stepping.

We then compute the pressure difference ``\Delta P`` by solving

```math
L \Delta P = \frac{M V + y_M(t)}{\Delta t} - M (y_G(t) - y_G(t_0)),
```

after which a divergence free velocity ``U`` can be enforced:

```math
U = V - \Delta t \Omega_h^{-1} (G \Delta P + y_G(t) - y_G(t_0)).
```

A first order accurate prediction of the corresponding pressure is ``P = P_0 +
\Delta P``. However, since this pressure is reused in the next time step, we
perform an additional pressure solve to avoid accumulating first order errors.
The resulting pressure ``P`` is then accurate to the same order as ``U``.


## One-leg beta method

See Verstappen and Veldman [Verstappen2003](@cite) [Verstappen1997](@cite).

We here require that the time step ``\Delta t`` is constant. Given the velocity
``U_0`` and pressure ``P_0`` at the current time ``t_0`` and their previous
values ``U_{-1}`` and ``P_{-1}`` at the time ``t_{-1} = t_0 - \Delta t``, we
start by computing the "offstep" values ``V = (1 + \beta) V_0 - \beta V_{-1}``
and ``Q = (1 + \beta) P_0 - \beta P_{-1}`` for some ``\beta = \frac{1}{2}``.

A tentative velocity field ``W`` is then computed as follows:

```math
W = \frac{1}{\beta + \frac{1}{2}} \left( 2 \beta U_0 - \left( \beta -
\frac{1}{2} \right) U_{-1} + \Delta t \Omega_h^{-1} F(V, t) - \Delta t
\Omega_h^{-1} (G Q + y_G(t)) \right).
```

A pressure correction ``\Delta P `` is obtained by solving the Poisson equation
```math
L \Delta P = \frac{\beta + \frac{1}{2}}{\Delta t} (M W + y_M(t)).
```

Finally, the divergence free velocity field is given by

```math
U = W - \frac{\Delta t}{\beta + \frac{1}{2}} \Omega_h^{-1} G \Delta P,
```

while the second order accurate pressure is given by

```math
P = 2 P_0 - P_{-1} + \frac{4}{3} \Delta P.
```
