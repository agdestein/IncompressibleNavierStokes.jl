# Spatial Discretization

The incompressible Navier-Stokes equations are given by

```math
\begin{align*}
\nabla \cdot u & = 0, \\
\frac{\mathrm{d} u}{\mathrm{d} t} + u \cdot \nabla u & = -\nabla p +
\nu \nabla^2 u + f.
\end{align*}
```

where ``u`` is the velocity field, ``p`` is the pressure, ``\nu`` is
the kinematic viscosity, and ``f`` is the body force.

All discrete operators are represented as sparse matrices.

The discretized Navier-Stokes equations are given by

```math
\begin{align*}
M V(t) & = y_M \\
\Omega \frac{\mathrm{d} V}{\mathrm{d} t}(t) & = -C(V(t)) V(t) + \nu (D V(t) +
y_D) + f - (G p(t) + y_G)
\end{align*}
```

For more information on the discretization, see [Sanderse2012](@cite),
[Sanderse2013](@cite), [Sanderse2014](@cite).
