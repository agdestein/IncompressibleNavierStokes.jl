# Spatial Discretization

To discretize the incompressible Navier-Stokes equations, we will use finite
volumes on a staggered Cartesian grid, as proposed by Harlow and Welsh
[Harlow1965](@cite). We will use the notation of Sanderse [Sanderse2012](@cite)
[Sanderse2013](@cite) [Sanderse2014](@cite).

Let ``d \in \{2, 3\}`` denote the spatial dimension (2D or 3D). We will make
use of the "Cartesian" index ``I = (i, j)`` in 2D or ``I = (i, j, k)`` in 3D,
with ``I(1) = i``, ``I(2) = j``, and ``I(3) = k``. Here, the indices ``I``,
``i``, ``j``, and ``k``, represent discrete degrees of freedom. To specify a
spatial dimension, we will use the symbols ``(\alpha, \beta, \gamma) \in \{1,
\dots, d\}^3``. We will use the symbol ``\delta(\alpha) = (\delta_{\alpha
\beta})_{\beta = 1}^d \in \{0, 1\}^d`` to indicate a perturbation in the
direction ``\alpha``, where ``\delta_{\alpha \beta}`` is the Kronecker symbol.
The spatial variable is ``x = (x^1, \dots, x^d) \in \Omega \subset
\mathbb{R}^d``. Note that ``\Omega = \prod_{\alpha = 1}^d [0, L^\alpha]`` is
assumed to have the shape of a box with side lengths ``L^\alpha > 0``.

## Finite volumes

The finite volumes are defined as

```math
\Omega_I = \prod_{\alpha = 1}^d \left[ x^\alpha_{I(\alpha) - \frac{1}{2}},
x^\alpha_{I(\alpha) + \frac{1}{2}} \right], \quad I \in \mathcal{I}.
```

They represent rectangles in 2D and prisms in 3D. They are fully defined by the
vectors of volume faces ``x^\alpha = \left( x^\alpha_{i + \frac{1}{2}}
\right)_{i = 0}^{N(\alpha)}``, where ``N = (N(1), \dots, N(d)) \in
\mathbb{N}^d`` are the numbers of volumes in each dimension and ``\mathcal{I} =
\prod_{\alpha = 1}^d \{1, \dots, N(\alpha)\}`` the set of finite volume
indices. Note that the components ``x^\alpha_{i}`` are not assumed to be
uniformly spaced. But we do assume that they are strictly increasing with
``i``.

The volume center coordinates are determined from the volume boundaries by
``x^\alpha_{i} = \frac{1}{2} (x^\alpha_{i - \frac{1}{2}} + x_{i +
\frac{1}{2}})``. This allows for defining the shifted volumes ``\Omega_{I +
\delta(\alpha) / 2}``.

We also define the volume widths/depths/heights ``\Delta x^\alpha_i =
x^\alpha_{i + \frac{1}{2}} - x^\alpha_{i - \frac{1}{2}}``, where ``i`` can take
half values. The volume sizes are thus ``| \Omega_{I} | = \prod_{\alpha = 1}^d
\Delta x^\alpha_{I(\alpha)}``.

In addition to the finite volumes and their half-indexed shifted variants, we
define the surface

```math
\Gamma^\alpha_I = \prod_{\beta = 1}^d \begin{cases}
    \left\{ x^\beta_{I(\beta)} \right\}, & \quad \alpha = \beta \\
    \left[ x^\beta_{I(\beta) - 1 / 2}, x^\beta_{I(\beta) + 1 / 2} \right], & \quad
    \text{otherwise},
\end{cases}
```
where ``I`` can take half-values. It is the interface between ``\Omega_{I -
\delta(\alpha) / 2}`` and ``\Omega_{I + \delta(\alpha) / 2}``, and has surface
normal ``\delta(\alpha)``.


In each finite volume ``\Omega_{I}`` (integer ``I``), there are three different
types of positions in which quantities of interest can be defined:

- The volume center ``x_I = (x_{I(1)}, \dots, x_{I(d)})``, where the discrete
  pressure ``p_I`` is defined;
- The right/rear/top volume face centers ``x_{I + \delta(\alpha) / 2}``, where
  the discrete ``\alpha``-velocity component ``u^\alpha_{I + \delta(\alpha) / 2}`` is defined;
- The right-rear-top volume corner  ``x_{I + \sum_{\alpha} \delta(\alpha) /
  2}``, where the discrete vorticity ``\omega_{I + \sum_{\alpha} \delta(\alpha) /
  2}`` is defined.

The vectors of unknowns ``u^\alpha_h`` and ``p_h`` will not contain all the
half-index components, only those from their own canonical position.

!!! note "Storage convention"
    We use the column-major convention (Julia, MATLAB, Fortran), and not the
    row-major convention (Python, C). Thus the ``x^1``-index ``i`` will vary for
    one whole cycle in the vectors before the
    ``x^2``-index ``j`` is incremented, e.g. ``p_h = (p_{(1, 1, 1)},
    p_{(2, 1, 1)}, \dots p_{(N(1), N(2), N(3))})`` in 3D.

In 2D, this finite volume configuration is illustrated as follows:

![Grid](../assets/grid.png)

## Interpolation

When a quantity is required *outside* of its native point, we will use interpolation. Examples:

- To compute ``u^\alpha`` at the pressure point ``x_I``:
  ```math
  \begin{split}
      u^\alpha_I & = \frac{x^\alpha_{I(\alpha) + 1 / 2} -
      x^\alpha_{I(\alpha)}}{x^\alpha_{I(\alpha) + 1 / 2} - x^\alpha_{I(\alpha) - 1 / 2}}
      u_{I - \delta(\alpha) / 2}
      + \frac{x^\alpha_{I(\alpha)} - x^\alpha_{I(\alpha) - 1 / 2}}{x^\alpha_{I(\alpha) + 1 / 2} - x^\alpha_{I(\alpha) - 1 / 2}}
      u_{I + \delta(\alpha) / 2} \\
      & = \frac{1}{2} \left( u_{I - \delta(\alpha) / 2} + u_{I + \delta(\alpha) / 2} \right).
  \end{split}
  ```
  Interpolation weights from volume faces to volume centers are always
  ``\frac{1}{2}``.
- To compute ``u^\alpha`` at center of edge between ``\alpha``-face and
  ``\beta``-face ``x_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}``:
  ```math
  u^\alpha_{I + \delta(\alpha) / 2 + \delta(\beta) / 2} =
  \frac{x^\beta_{I(\beta) + 1} - x^\beta_{I(\beta) + 1 / 2}}{x^\beta_{I(\beta) + 1} - x^\beta_{I(\beta)}}
  u^\alpha_{I + \delta(\alpha) / 2}
  + \frac{x^\beta_{I(\beta) + 1 / 2} - x^\beta_{I(\beta)}}{x^\beta_{I(\beta) + 1} - x^\beta_{I(\beta)}}
  u^\alpha_{I + \delta(\alpha) / 2 + \delta(\beta)}.
  ```
  Note that the grid is allowed to be non-uniform, so the interpolation weights
  may unequal and different from ``\frac{1}{2}``.
- To compute ``p`` at ``u^\alpha``-points:
  ```math
  p_{I + \delta(\alpha) / 2} =
  \frac{x^\alpha_{I(\alpha) + 1} - x^\alpha_{I(\alpha) + 1 / 2}}{x^\alpha_{I(\alpha) + 1} - x^\alpha_{I(\alpha)}}
  p_{I}
  + \frac{x^\alpha_{I(\alpha) + 1 / 2} - x^\alpha_{I(\alpha)}}{x^\alpha_{I(\alpha) + 1} - x^\alpha_{I(\alpha)}}
  p_{I + \delta(\alpha)}
  ```

## Finite volume discretization of the Navier-Stokes equations

We will consider the integral form of the Navier-Stokes equations. This has the
advantage that some of the spatial derivatives dissapear, reducing the amount
of finite difference approximations we need to perform.

### Mass equation

The mass equation takes the form

```math
\int_{\partial \mathcal{O}} u \cdot n \, \mathrm{d} \Gamma = 0, \quad \forall
\mathcal{O} \subset \Omega.
```

Using the pressure volume ``\mathcal{O} = \Omega_{I}``, we get

```math
\sum_{\alpha = 1}^d \left( \int_{\Gamma^\alpha_{I + \delta(\alpha) / 2}} u^\alpha \, \mathrm{d} \Gamma -
\int_{\Gamma_{I - \delta(\alpha) / 2}^\alpha} u^\alpha \, \mathrm{d} \Gamma
\right) = 0.
```

Assuming that the flow is fully resolved, meaning that ``\Omega_{I}`` is is
sufficiently small such that ``u`` is locally linear, we can perform the
local approximation (quadrature)

```math
\int_{\Gamma^\alpha_I} u^\alpha \, \mathrm{d} \Gamma \approx | \Gamma^\alpha_I | u^\alpha_{I}.
```

This yields the discrete mass equation

```math
\sum_{\alpha = 1}^d
\left| \Gamma^\alpha_I \right| \left( u^\alpha_{I + \delta(\alpha) / 2} -
u^\alpha_{I - \delta(\alpha) / 2} \right) = 0
```

which can also be written in the matrix form

```math
M V = \sum_{\alpha = 1}^d M^\alpha u^\alpha = 0,
```

where ``M = \begin{pmatrix} M_1 & \dots & M_d \end{pmatrix}`` is the discrete
divergence operator.

!!! note "Approximation error"
    For the mass equation, the only approximation we have performed is
    quadrature. No interpolation or finite difference error is present.

### Momentum equations

Grouping the convection, pressure gradient, diffusion, and body force terms in
each of their own integrals, we get, for all ``\mathcal{O} \subset \Omega``:

```math
\frac{\partial }{\partial t} \int_\mathcal{O} u^\alpha \, \mathrm{d} \Omega
=
- \sum_{\beta = 1}^d \int_{\partial \mathcal{O}} u^\alpha u^\beta n^\beta \, \mathrm{d} \Gamma
- \int_{\partial \mathcal{O}} p n^\alpha \, \mathrm{d} \Gamma
+ \nu \sum_{\beta = 1}^d \int_{\partial \mathcal{O}} \frac{\partial u^\alpha}{\partial x^\beta} n^\beta \, \mathrm{d} \Gamma
+ \int_\mathcal{O} f^\alpha \mathrm{d} \Omega,
```

where ``n = (n^1, \dots, n^d)`` is the surface normal vector to ``\partial
\Omega``.

This time, we will not let ``\mathcal{O}`` be the reference finite volume
``\Omega_{I}`` (the ``p``-volume), but rather the shifted ``u^\alpha``-volume.
Setting ``\mathcal{O} = \Omega_{I + \delta(\alpha) / 2}`` (with right/rear/top
``\beta``-faces ``\Gamma^\beta_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}``)
gives

```math
\begin{split}
    \frac{\partial }{\partial t}
    \int_{\Omega_{I + \delta(\alpha) / 2}}
    \! \! \! 
    \! \! \! 
    \! \! \! 
    \! \! \! 
    u^\alpha \, \mathrm{d} \Omega
    =
    & -
    \sum_{\beta = 1}^d \left(
        \int_{\Gamma^{\beta}_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}}
            \! \! \! 
            \! \! \! 
            \! \! \! 
            \! \! \! 
        u^\alpha u^\beta \, \mathrm{d} \Gamma 
        - \int_{\Gamma^{\beta}_{I + \delta(\alpha) / 2 - \delta(\beta) / 2}}
            \! \! \! 
            \! \! \! 
            \! \! \! 
            \! \! \! 
            \! \! \! 
            \! \! \! 
        u^\alpha u^\beta \, \mathrm{d} \Gamma 
    \right) \\
    & -
    \left(
        \int_{\Gamma^{\alpha}_{I + \delta(\alpha)}} p \, \mathrm{d} \Gamma
    - \int_{\Gamma^{\alpha}_{I}} p \, \mathrm{d} \Gamma
    \right) \\
    & + \nu \sum_{\beta = 1}^d 
    \left(
        \int_{\Gamma^{\beta}_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}}
        \frac{\partial u^\alpha}{\partial x^\beta} \, \mathrm{d} \Gamma 
        - \int_{\Gamma^{\beta}_{I + \delta(\alpha) / 2 - \delta(\beta) / 2}}
        \frac{\partial u^\alpha}{\partial x^\beta} \, \mathrm{d} \Gamma 
    \right) \\
    & +
    \int_{\Omega_{I + \delta(\alpha) / 2}}
    f^\alpha \, \mathrm{d} \Omega
\end{split}
```

This equation is still exact. We now introduce some approximations on
``\Omega_{I + \delta(\alpha) / 2}`` and its boundaries to remove all unknown
continuous quantities.

1. We replace the integrals with a mid-point quadrature rule.
1. The mid-point values of derivatives are approximated using a central-like finite difference:
   ```math
   \frac{\partial u^\alpha}{\partial x^\beta}(x_I) \approx
   \frac{u^\alpha_{I + \delta(\beta) / 2}
   - u^\alpha_{I - \delta(\beta) / 2}}{x^\beta_{I(\beta) + 1 / 2} - x^\beta_{I(\beta) - 1 / 2}}.
   ```
1. Quantities outside their canonical positions are obtained through
   interpolation.

Finally, the discrete ``\alpha``-momentum equations are given by

```math
\begin{split}
    \left| \Omega_{I + \delta(\alpha) / 2} \right|
    \frac{\mathrm{d} }{\mathrm{d} t} u^\alpha_{I + \delta(\alpha) / 2} =
    & - \sum_{\beta = 1}^d \left| \Gamma^\beta_{I + \delta(\alpha) / 2} \right|
    \left(
        (u^\alpha
        u^\beta)_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}
        -
        (u^\alpha
        u^\beta )_{I + \delta(\alpha) / 2 - \delta(\beta) / 2}
    \right) \\
    & - \left| \Gamma^\alpha_{I + \delta(\alpha) / 2} \right|
    \left( p_{I + \delta(\alpha)} - p_{I} \right) \\
    & + \nu \sum_{\beta = 1}^d \left| \Gamma^\beta_{I + \delta(\alpha) / 2} \right|
    \left( 
        \frac{u^\alpha_{I + \delta(\alpha) / 2 + \delta(\beta)} - u^\alpha_{I +
        \delta(\alpha) / 2}}{x^\beta_{I(\beta) + 1} - x^\beta_{I(\beta)}}
        - \frac{u^\alpha_{I + \delta(\alpha) / 2} - u^\alpha_{I + \delta(\alpha)
        / 2 - \delta(\beta)}}{x^\beta_{I(\beta)} - x^\beta_{I(\beta) - 1}}
    \right) \\
    & + \left| \Omega_{I + \delta(\alpha) / 2} \right| f^\alpha(x_{I + \delta(\alpha) / 2}).
\end{split}
```

In matrix form, we will denote this as

```math
\Omega_h \frac{\mathrm{d} V_h}{\mathrm{d} t} = - C(V_h) + \nu D V_h + \Omega_h f_h  - G p_h.
```

Note the important property ``G = M^\mathsf{T}``.


## Boundary conditions

If a domain boundary is not periodic, the boundary values of certain quantities
are prescribed. Consider the left boundary defined by ``i = \frac{1}{2}``.

- For Dirichlet boundary conditions, we prescribe the value
  ```math
  u_{\frac{1}{2}, j} := u(x_{\frac{1}{2}}, y_j)
  ```
  For ``v``, we also prescribe
  values, but only at the boundary, thus replacing otherwise interpolated
  ``v``-fluxes:
  ```math
  v_{\frac{1}{2}, j - \frac{1}{2}} := v(x_{\frac{1}{2}}, y_{j - \frac{1}{2}}).
  ```
- For a symmetric left boundary, only ``u_{\frac{1}{2}, j}`` is prescribed.
- For a pressure left boundary, we prescribe ``p``:
  ```math
  p_{0, j} = p(x_0, y_j).
  ```

The boundary components are removed from the two vectors ``V_h`` and ``p_h``.
Instead, these components are prescribed as constants.

The discrete mass equation then becomes
```math
M V_h = y_M,
```
where ``y_M`` are the boundary conditions for ``M``.

The discrete momentum equations become

```math
\begin{split}
    \Omega_h \frac{\mathrm{d} V_h}{\mathrm{d} t} & = -C(V_h) V_h + \nu (D V_h +
    y_D) + f_h - (G p_h + y_G) \\
    & = F(V_h) - (G p_h + y_G),
\end{split}
```

where ``y_D`` is diffusion boundary vector and ``y_G`` is the
pressure boundary vector.


## Discrete pressure Poisson equation

Instead of directly discretizing the continuous pressure Poisson equation, we
will rededuce it in the *discrete* setting, thus aiming to preserve the
discrete divergence freeness instead of the continuous one. Applying the
discrete divergence operator ``M`` to the discrete momentum equations yields
the discrete pressure Poisson equation

```math
- L p_h = - M \Omega_h^{-1} (F(V_h) - y_G) + \frac{\mathrm{d} y_M}{\mathrm{d} t}
```

where ``L = M \Omega_h^{-1} G`` is a discrete Laplace operator. It is positive
symmetric since ``G = M^\mathsf{T}``.


!!! note "Pressure projection"
    The pressure vector ``p_h`` can be seen as a Lagrange multiplier enforcing
    the constraint of divergence freeness. It is possible to write a the
    momentum equations without the pressure by explicitly solving the discrete
    Poisson equation:

    ```math
    p_h = L^{-1} M \Omega_h^{-1} (F(V_h) - y_G) - L^{-1} \frac{\mathrm{d} y_M}{\mathrm{d} t}.
    ```

    The momentum equations then become

    ```math
    \Omega_h \frac{\mathrm{d} V_h}{\mathrm{d} t} = (I - G L^{-1} M \Omega_h^{-1})
    (F(V_h) - y_G) + G L^{-1} \frac{\mathrm{d} y_M}{\mathrm{d} t}.
    ```

    The matrix ``(I - G L^{-1} M \Omega^{-1})`` is a projector onto the space
    of discretely divergence free velocities. However, using this formulation
    would require an efficient way to perform the projection without assembling
    the operator matrix ``L^{-1}``, which would be very costly.
