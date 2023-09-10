# Time discretization

The spatially discretized Navier-Stokes equations form a differential-algebraic
system, with an ODE for the velocity

```math
\Omega \frac{\mathrm{d} u_h}{\mathrm{d} t} = F(u_h) - (G p_h + y_G)
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

Consider the interval ``[0, T]``. for some simulation time ``T``. We will
divide it into ``N`` sub-intervals ``[t^n, t^{n + 1}]`` for ``n = 0, \dots, N -
1``, with ``t^0 = 0``, ``t^N = T``, and increment ``\Delta t^n = t^{n + 1} -
t^n``. We define ``U^n \approx u_h(t_n)`` as an approximation to the exact
discrete velocity field ``u_h(t_n)``, with ``U^0 = u_h(0)`` starting from the
exact initial conditions.

## Explicit Runge-Kutta methods

See Sanderse [Sanderse2012](@cite).

For explicit Runge-Kutta methods, we divide each time step into ``s`` sub-steps.
For ``i = 1, \dots, s``, we define the following quantities:

```math
\begin{split}
U_0^n & = U^n \\
\Delta t_i^n & = c_i \Delta t^n \\
t_i^n & = t^n + \Delta t_i^n \\
F_i & = F(U_{i - 1}^n, t_{i - 1}^n) \\
V_i^n & = U^n + \Delta t^n \sum_{j = 1}^i a_{i j} F_j \\
L P_i^n & = \frac{(M V_i^n + y_M(t_i^n)) - (M U^n + y_M(t^n))}{\Delta t_i^n} \\
& = \frac{1}{c_i} \sum_{j = 1}^i a_{i j} F_j +
\frac{y_M(t_i^n) - y_M(t^n)}{\Delta t_i^n} \\
U_i^n & = V_i^n - \Delta t_i^n G P_i^n,
\end{split}
```

where ``(a_ij)_{i j}`` are the Butcher tableau coefficients of the RK-method,
with the convention ``c_i = \sum_{j = 1}^i a_{i j}`` and ``c_0 = 0``.

Finally, we set ``U^{n + 1} = U_s^n``. The corresponding pressure ``P^{n + 1}``
can be calculated to the same accuracy as ``U^{n + 1}`` by doing an additional
pressure projection (if we know ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t^{n +
1})``), or to first order accuracy by simply taking ``P^{n + 1} = P_s^n``.

Note that each of the sub-step velocities ``U_i^n`` is divergence free, after
projecting the tentative velocities ``V_i^s``. This is ensured due to the
judiciously chosen replacement of ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t_i^n)`` with
``(y_M(t_i^n) - y_M(t^n)) / \Delta t_i^n``.

## Implicit Runge-Kutta methods

See Sanderse [Sanderse2013](@cite).

## Adams-Bashforth Crank-Nicolson method

## One-leg beta method

