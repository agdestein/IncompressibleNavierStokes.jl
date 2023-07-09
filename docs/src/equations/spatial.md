# Spatial Discretization

To discretize the incompressible Navier-Stokes equations, we will use finite
volumes on a staggered grid. This was originally proposed by Harlow and Welsh
[^1]. We will use the notation of Sanderse [^2] [^3] [^4]. For simplicity, we
will illustrate everything in 2D. The 3D discretization is very similar, but
significantly more verbose.

The finite volumes are given by ``\Omega_{i, j} = [x_{i -
\frac{1}{2}}, x_{i + \frac{1}{2}}] \times [y_{j - \frac{1}{2}}, y_{j +
\frac{1}{2}}]``. They are fully defined by the vectors of volume faces ``x =
(x_{i + \frac{1}{2}})_i`` and ``y = (y_{j + \frac{1}{2}})_j``. Note that the
components are not assumed to be uniformly spaced. But we do assume that they
are strictly increasing.

The volume center coordinates are deduced as ``x_i = \frac{1}{2} (x_{i -
\frac{1}{2}} + x_{i + \frac{1}{2}})`` and ``y_j = \frac{1}{2} (y_{j -
\frac{1}{2}} + y_{j + \frac{1}{2}})``.

- The ``u``-point ``(x_{i + \frac{1}{2}}, y_j)``,
- The ``v``-point ``(x_i, y_{j + \frac{1}{2}})``,
- The ``p``-point ``(x_i, y_j)``,
- The ``\omega``-point ``(x_{i + \frac{1}{2}}, y_{j + \frac{1}{2}})``.

## Interpolation

The vectors of unknowns will not contain all the half-index values:

- The vector ``u_h`` will only consist of the components ``u_{i + \frac{1}{2}, j}``.
- The vector ``v_h`` will only consist of the components ``v_{i, j + \frac{1}{2}}``.
- The vector ``p_h`` will only consist of the components ``p_{i, j}``.

When components not contained in the vector of unknowns are needed to compute
derivatives, we will use averaging to compute volume-center values (integer ``i``
and ``j``), and interpolation to compute volume corner/face values
(half-index for ``i`` and/or ``j`` respectively). Examples:

- To compute ``u`` at the pressure points: Averaging (interpolation simplifies)
  ```math
  \begin{split}
      u_{i, j} & =
      \frac{x_{i + \frac{1}{2}} - x_i}{x_{i + \frac{1}{2}} - x_{i - \frac{1}{2}}}
      u_{i - \frac{1}{2}, j}
      + \frac{x_i - x_{i - \frac{1}{2}}}{x_{i + \frac{1}{2}} - x_{i - \frac{1}{2}}}
      u_{i + \frac{1}{2}, j} \\
      & = 
      \frac{1}{2} (u_{i - \frac{1}{2}, j} + u_{i + \frac{1}{2}, j})
  \end{split}
  ```
  Interpolation weights from volume faces to volume centers are always
  ``\frac{1}{2}``.
- To compute ``v`` at vorticity points: Interpolation
  ```math
  v_{i + \frac{1}{2}, j + \frac{1}{2}} =
  \frac{x_{i + 1} - x_{i + \frac{1}{2}}}{x_{i + 1} - x_i}
  v_{i, j + \frac{1}{2}}
  + \frac{x_{i + \frac{1}{2}} - x_i}{x_{i + 1} - x_i}
  v_{i + 1, j + \frac{1}{2}}
  ```
- To compute ``p`` at ``v``-points: Interpolation
  ```math
  p_{i, j + \frac{1}{2}} =
  \frac{y_{j + 1} - y_{j + \frac{1}{2}}}{y_{j + 1} - y_j}
  p_{i, j}
  + \frac{y_{j + \frac{1}{2}} - y_j}{y_{j + 1} - y_j}
  p_{i, j + 1}
  ```

Note that the grid is allowed to be non-uniform, so the weights of interpolation from
volume centers to volume faces may unequal and different from ``\frac{1}{2}``.

## Differential operators

Consider a scalar field ``q`` defined at all the four points of each volume. The
exact quantity ``\frac{\partial q}{\partial x}(x_i, y_j)`` is
approximated by the difference ``q_{i, j}^x`` given by

```math
q_{i, j}^x =
\frac{q_{i + \frac{1}{2}, j} - q_{i -
\frac{1}{2}, j}}{x_{i + \frac{1}{2}} - x_{i - \frac{1}{2}}},
```

where ``i`` and ``j`` can take integer or half values. If ``q`` is not
available at the required points, interpolation is performed *before*
differentiation takes place.

Similarly, ``\frac{\partial q}{\partial y}(x_i, y_j)`` is
always discretized as

```math
q_{i, j}^y = \frac{q_{i, j + \frac{1}{2}} - q_{i, j -
\frac{1}{2}}}{y_{j + \frac{1}{2}} - y_{j - \frac{1}{2}}}.
```

These central differences are second order accurate in ``(x_{i + \frac{1}{2}} - x_{i - \frac{1}{2}})`` and ``(y_{j + \frac{1}{2}} - y_{j - \frac{1}{2}})`` respectively.

The Laplace operator is obtained by applying each of the first order operators
twice:

```math
q_{i, j}^{x x} =
\frac{q_{i + \frac{1}{2}, j}^x - q_{i -
\frac{1}{2}, j}^x}{x_{i + \frac{1}{2}} - x_{i - \frac{1}{2}}},
```

and

```math
q_{i, j}^{y y} = \frac{q_{i, j + \frac{1}{2}}^y - q_{i, j -
\frac{1}{2}}^y}{y_{j + \frac{1}{2}} - y_{j - \frac{1}{2}}}.
```


## Discretizing the Navier-Stokes equations

We start by discretize
the mass equation such that we have an exact definition for the discrete
divergence freeness. The divergence operator defining this discrete equation is
defined in the ``p``-points. Using the above operators, this gives the
following equation in each pressure point:

```math
\frac{u_{i + \frac{1}{2}, j} - u_{i - \frac{1}{2}, j}}{x_{i + \frac{1}{2}} -
x_{i - \frac{1}{2}}} +
\frac{v_{i, j + \frac{1}{2}} - v_{i, j - \frac{1}{2}}}{y_{j + \frac{1}{2}} -
y_{j - \frac{1}{2}}}
= 0
```

We then discretize the two momentum equations, making sure that

- the ``u``-momentum equation is defined at the ``u``-points,
- the ``v``-momentum equation is defined at the ``v``-points.

The discrete momentum equations thus

```math
```

Instead of directly discretizing the continuous pressure Poisson equation, we
will rededuce it in the *discrete* setting, thus aiming to preserve the
discrete divergence freeness instead of the continuous one. Taking the discrete
divergence of the to momentum equations yield the discrete pressure Poisson
equation (defined at the ``p``-points):

```math
```

## Boundary conditions

## Operator formulation

All discrete operators are represented as sparse matrices.

The discretized Navier-Stokes equations are given by

```math
\begin{align*}
M V(t) & = y_M \\
\Omega \frac{\mathrm{d} V}{\mathrm{d} t}(t) & = -C(V(t)) V(t) + \nu (D V(t) +
y_D) + f - (G p(t) + y_G)
\end{align*}
```

[^1]: [Harlow1965](@cite)
[^2]: [Sanderse2014](@cite)
[^3]: [Sanderse2013](@cite)
[^4]: [Sanderse2014](@cite)
