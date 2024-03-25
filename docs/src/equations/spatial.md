# Spatial Discretization

To discretize the incompressible Navier-Stokes equations, we will use finite
volumes on a staggered Cartesian grid, as proposed by Harlow and Welsh
[Harlow1965](@cite). We will use the notation of Sanderse [Sanderse2012](@cite)
[Sanderse2013](@cite) [Sanderse2014](@cite).

Let ``d \in \{2, 3\}`` denote the spatial dimension (2D or 3D). We will make
use of the "Cartesian" index ``I = (i, j)`` in 2D or ``I = (i, j, k)`` in 3D,
with ``I(1) = i``, ``I(2) = j``, and ``I(3) = k``. Here, the indices ``I``,
``i``, ``j``, and ``k``, represent discrete degrees of freedom, and can take
integer or half values (e.g. ``3`` or ``5/2``). To specify a
spatial dimension, we will use the symbols ``(\alpha, \beta, \gamma) \in \{1,
\dots, d\}^3``. We will use the symbol ``\delta(\alpha) = (\delta_{\alpha
\beta})_{\beta = 1}^d \in \{0, 1\}^d`` to indicate a perturbation in the
direction ``\alpha``, where ``\delta_{\alpha \beta}`` is the Kronecker symbol.
The spatial variable is ``x = (x^1, \dots, x^d) \in \Omega \subset
\mathbb{R}^d``.


## Finite volumes

We here assume that ``\Omega = \prod_{\alpha = 1}^d [0, L^\alpha]`` has the
shape of a box with side lengths ``L^\alpha > 0``. This allows for partitioning
``\Omega`` into the finite volumes

```math
\Omega_I = \prod_{\alpha = 1}^d \left[ x^\alpha_{I(\alpha) - \frac{1}{2}},
x^\alpha_{I(\alpha) + \frac{1}{2}} \right], \quad I \in \mathcal{I}.
```

Just like ``\Omega`` itself, they represent rectangles in 2D and prisms n 3D.
They are fully defined by the vectors of volume face coordinates ``x^\alpha =
\left( x^\alpha_{i} \right)_{i = 0}^{N(\alpha)} \in \mathbb{R}^{N(\alpha) +
1}``, where ``N = (N(1), \dots, N(d)) \in \mathbb{N}^d`` are the numbers of
finite volumes in each dimension and ``\mathcal{I} = \prod_{\alpha = 1}^d
\left\{ \frac{1}{2}, 2 - \frac{1}{2}, \dots, N(\alpha) - \frac{1}{2} \right\}``
the set of finite volume indices (note that the reference volumes are indexed
by half indices only). The components ``x^\alpha_{i}`` are not assumed to be
uniformly spaced. But we do assume that they are strictly increasing with
``i``, with ``x^\alpha_0 = 0`` and ``x^\alpha_{N(\alpha)} = L^\alpha``.

The coordinates of the volume centers are determined from the those of the
volume boundaries by
``x^\alpha_{I(\alpha)} = \frac{1}{2} (x^\alpha_{I(\alpha) - \frac{1}{2}} +
x_{I(\alpha) + \frac{1}{2}})``
for ``I \in \mathcal{I}``. This allows for defining shifted volumes such as
``\Omega_{I + \delta(\alpha) / 2}`` and
``\Omega_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}``. The original volumes
(with indices in ``\mathcal{I}``) will be referred to as the reference finite
volumes.

We also define the volume widths/depths/heights ``\Delta x^\alpha_i =
x^\alpha_{i + \frac{1}{2}} - x^\alpha_{i - \frac{1}{2}}``, where ``i`` can take
half values. The volume sizes are thus ``| \Omega_{I} | = \prod_{\alpha = 1}^d
\Delta x^\alpha_{I(\alpha)}``.
In addition to the finite volumes and their shifted variants, we
define the interface
``\Gamma^\alpha_I = \Omega_{I - \delta(\alpha) / 2} \cup \Omega_{I +
\delta(\alpha) / 2}``.

In each reference finite volume ``\Omega_{I}`` (``I \in \mathcal{I}``), there
are three different types of positions in which quantities of interest can be
defined:

- The volume center ``x_I = (x_{I(1)}, \dots, x_{I(d)})``, where the discrete
  pressure ``p_I`` is defined;
- The right/rear/top volume face centers ``x_{I + \delta(\alpha) / 2}``, where
  the discrete ``\alpha``-velocity component ``u^\alpha_{I + \delta(\alpha) / 2}`` is defined;
- The right-rear-top volume corner  ``x_{I + \sum_{\alpha} \delta(\alpha) /
  2}``, where the discrete vorticity ``\omega_{I + \sum_{\alpha} \delta(\alpha) /
  2}`` is defined.

The vectors of unknowns ``u^\alpha_h`` and ``p_h`` will not contain all the
half-index components, only those from their own canonical position. The
unknown discrete pressure represents the average pressure in each reference
volume, and the unknown discrete velocity components represent exchange of mass
between neighboring volumes.

In 2D, this finite volume configuration is illustrated as follows:

![Grid](../assets/grid.png)

## Interpolation

When a quantity is required *outside* of its native point, we will use interpolation. Examples:

- To compute ``u^\alpha`` at the pressure point ``x_I``, ``I \in \mathcal{I}``:
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
advantage that some of the spatial derivatives disappear, reducing the amount
of finite difference approximations we need to perform.

We define the finite difference operator ``\partial_\alpha`` equivalent to
the continuous operator ``\frac{\partial}{\partial x^\alpha}``. For all fields
discrete fields ``\varphi``, it is given by

```math
(\partial_\alpha \varphi)_I = \frac{\varphi_{I + \delta(\alpha) / 2} - \varphi_{I -
\delta(\alpha) / 2}}{\Delta^\alpha_{I(\alpha)}},
```

where ``\varphi`` is interpolated first if necessary.


### Mass equation

The mass equation takes the form

```math
\frac{1}{| \mathcal{O} |}
\int_{\partial \mathcal{O}} u \cdot n \, \mathrm{d} \Gamma = 0,
\quad \forall \mathcal{O} \subset \Omega.
```

Using the pressure volume ``\mathcal{O} = \Omega_{I}``, we get

```math
\sum_{\alpha = 1}^d
\frac{1}{| \Omega_I |}
\left( \int_{\Gamma^\alpha_{I + \delta(\alpha) / 2}}
u^\alpha \, \mathrm{d} \Gamma - \int_{\Gamma_{I - \delta(\alpha) / 2}^\alpha}
u^\alpha \, \mathrm{d} \Gamma
\right) = 0.
```

Assuming that the flow is fully resolved, meaning that ``\Omega_{I}`` is is
sufficiently small such that ``u`` is locally linear, we can perform the
local approximation (quadrature)

```math
\int_{\Gamma^\alpha_I} u^\alpha \, \mathrm{d} \Gamma \approx | \Gamma^\alpha_I
| u^\alpha_{I}.
```

This yields the discrete mass equation

```math
\sum_{\alpha = 1}^d (\partial_\alpha u^\alpha)_{I} = 0.
```

!!! note "Approximation error"
    For the mass equation, the only approximation we have performed is
    quadrature. No interpolation or finite difference error is present.


### Momentum equations

Grouping the convection, pressure gradient, diffusion, and body force terms in
each of their own integrals, we get, for all ``\mathcal{O} \subset \Omega``:

```math
\begin{split}
\frac{\mathrm{d}}{\mathrm{d} t}
\frac{1}{| \mathcal{O} |}
\int_\mathcal{O} u^\alpha \, \mathrm{d} \Omega
=
& - \sum_{\beta = 1}^d
\frac{1}{| \mathcal{O} |}
\int_{\partial \mathcal{O}}
u^\alpha u^\beta n^\beta
\, \mathrm{d} \Gamma \\
& + \nu \sum_{\beta = 1}^d
\frac{1}{| \mathcal{O} |}
\int_{\partial \mathcal{O}}
\frac{\partial u^\alpha}{\partial x^\beta} n^\beta
\, \mathrm{d} \Gamma \\
& + \frac{1}{| \mathcal{O} |}
\int_\mathcal{O} f^\alpha \mathrm{d} \Omega \\
& - \frac{1}{| \mathcal{O} |}
\int_{\partial \mathcal{O}} p n^\alpha \, \mathrm{d} \Gamma,
\end{split}
```

where ``n = (n^1, \dots, n^d)`` is the surface normal vector to ``\partial
\mathcal{O}``.

This time, we will not let ``\mathcal{O}`` be the reference finite volume
``\Omega_{I}`` (the ``p``-volume), but rather the shifted ``u^\alpha``-volume.
Setting ``\mathcal{O} = \Omega_{I + \delta(\alpha) / 2}`` (with right/rear/top
``\beta``-faces ``\Gamma^\beta_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}``)
gives

```math
\begin{split}
    \frac{\mathrm{d}}{\mathrm{d} t}
    \frac{1}{| \Omega_{I + \delta(\alpha) / 2} |}
    \int_{\Omega_{I + \delta(\alpha) / 2}}
    \! \! \! 
    \! \! \! 
    \! \! \! 
    \! \! \! 
    u^\alpha \, \mathrm{d} \Omega
    =
    & -
    \sum_{\beta = 1}^d
    \frac{1}{| \Omega_{I + \delta(\alpha) / 2} |}
    \left(
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
    & + \nu \sum_{\beta = 1}^d 
    \frac{1}{| \Omega_{I + \delta(\alpha) / 2} |}
    \left(
        \int_{\Gamma^{\beta}_{I + \delta(\alpha) / 2 + \delta(\beta) / 2}}
        \frac{\partial u^\alpha}{\partial x^\beta} \, \mathrm{d} \Gamma 
        - \int_{\Gamma^{\beta}_{I + \delta(\alpha) / 2 - \delta(\beta) / 2}}
        \frac{\partial u^\alpha}{\partial x^\beta} \, \mathrm{d} \Gamma 
    \right) \\
    & +
    \frac{1}{| \Omega_{I + \delta(\alpha) / 2} |}
    \int_{\Omega_{I + \delta(\alpha) / 2}}
    f^\alpha \, \mathrm{d} \Omega \\
    & -
    \frac{1}{| \Omega_{I + \delta(\alpha) / 2} |}
    \left(
        \int_{\Gamma^{\alpha}_{I + \delta(\alpha)}} p \, \mathrm{d} \Gamma -
        \int_{\Gamma^{\alpha}_{I}} p \, \mathrm{d} \Gamma
    \right).
\end{split}
```

This equation is still exact. We now introduce some approximations on
``\Omega_{I + \delta(\alpha) / 2}`` and its boundaries to remove all unknown
continuous quantities.

1. We replace the integrals with a mid-point quadrature rule.
1. The mid-point values of derivatives are approximated using a central-like
   finite difference:
   ```math
   \frac{\partial u^\alpha}{\partial x^\beta}(x_I) \approx (\partial_\beta u^\alpha)_I
   ```
1. Quantities outside their canonical positions are obtained through
   interpolation.

Finally, the discrete ``\alpha``-momentum equations are given by

```math
\begin{split}
    \frac{\mathrm{d} }{\mathrm{d} t} u^\alpha_{I + \delta(\alpha) / 2} =
    - & \sum_{\beta = 1}^d
    (\partial_\beta (u^\alpha u^\beta))_{I + \delta(\alpha) / 2} \\
    + & \nu \sum_{\beta = 1}^d
    (\partial_\beta \partial_\beta u^\alpha)_{I + \delta(\alpha) / 2} \\
    + & f^\alpha(x_{I + \delta(\alpha) / 2}, t)
    - (\partial_\alpha p)_{I + \delta(\alpha) / 2}.
\end{split}
```

## Boundary conditions

Depending on the type of boundary conditions, certain indices used in the left-
and right hand sides of the equations may not be part of the solution vectors,
even if they are indeed at their canonical positions. Consider for example the
``\alpha``-left/front/bottom boundary ``\{ x \in \Omega | x^{\alpha} =
x^\alpha_{0} \}``. Let ``\Omega_I`` be one of the reference finite volumes
touching this boundary, i.e. ``I(\alpha) = \frac{1}{2}``.

- For periodic boundary conditions, we also consider the opposite boundary.
  We add two "ghost" reference volumes, one at each side of ``\Omega``:
    1. The ghost volume ``\Omega_{I - \delta(\alpha)}`` has the same shape as
       ``\Omega_{I + (N(\alpha) - 1) \delta(\alpha)}`` and contains the same
       components:
       - ``u^\alpha_{I - \delta(\alpha) / 2} = u^\alpha_{I + (N(\alpha) - 1 / 2)
         \delta(\alpha)}``,
       - ``u^\beta_{I + \delta(\beta) / 2 - \delta(\alpha)} =
         u^\beta_{I + \delta(\beta) / 2 + N(\alpha) \delta(\alpha)}`` for ``\beta
         \neq \alpha``,
       - ``p_{I - \delta(\alpha)} = p_{I + (N(\alpha) - 1) \delta(\alpha)}``.
    2. The ghost volume ``\Omega_{I + N(\alpha) \delta(\alpha)}`` has the same 
       shape as ``\Omega_I`` and contains the same pressure and velocity
       components.
- For Dirichlet boundary conditions, all the veloctity components are
  prescribed. For the normal velocity component, this is straightforward:
  ```math
  u^\alpha_{I - \delta(\alpha) / 2} = u^\alpha(x_{I - \delta(\alpha) / 2}).
  ```
  The parallel (``\beta \neq \alpha``) velocity components
  ``u^\beta_{I + \delta(\beta) / 2 - \delta(\alpha)}``
  appear in some of the right hand side expressions. Their ``\alpha``
  position ``x^\alpha_{- 1 / 2}`` has actually never been defined,
  so we simply define it to be on the boundary itself:
  ``x^\alpha_{- 1 / 2} = x^\alpha_0``. The value can then be
  prescribed:
  ```math
  u^\beta_{I + \delta(\beta) / 2 - \delta(\alpha)} = u^\beta \left( x_{I +
  \delta(\beta) / 2 - \delta(\alpha)} \right) = u^\beta \left( x_{I +
  \delta(\beta) / 2 - \delta(\alpha) / 2} \right).
  ```
  The pressure does not require any boundary modifications.
- For a symmetric boundary, the normal component has zero Dirichlet
  conditions, while the parallel components have zero Neumann conditions. For
  this, we add a ghost volume ``\Omega_{I - \delta(\alpha)}`` which has the
  same shape as ``\Omega_{I}``, meaning that ``x^\alpha_{-1} = x^\alpha_0 -
  (x^\alpha_1 - x^\alpha_0) = - x^\alpha_1``. The pressure in this volume is
  never used, but we set ``u^\alpha_{I - \delta(\alpha) / 2} = 0`` and
  ``u^\beta_{I + \delta(\beta) / 2 - \delta(\alpha)} = u^\beta_{I +
  \delta(\beta) / 2}`` for ``\beta \neq \alpha``.
- For a pressure boundary, the value of the pressure is prescribed, while the
  velocity has zero Neumann boundary conditions. For this, we add an
  infinitely thin ghost volume ``\Omega_{I - \delta(\alpha)}``, by setting
  ``x^\alpha_{-1} = x^\alpha_0 - \epsilon`` for some epsilon. We then let this
  thickness go to zero *after* substituting the boundary conditions in
  the momentum equations. The pressure is prescribed by
  ```math
  p_{I - \delta(\alpha)} = p(x_{I - \delta(\alpha)}) \underset{\epsilon \to 0}{\to} p(x_{I - \delta(\alpha) / 2}).
  ```
  The Neumann boundary conditions are obtained by setting
  ``u^\alpha_{I - 3 / 2 \delta(\alpha)} = u^\alpha_{I - \delta(\alpha) / 2}``
  and
  ``u^\beta_{I - \delta(\alpha) + \delta(\beta) / 2} = u^\beta_{I +
  \delta(\beta) / 2}`` for ``\beta \neq \alpha``.
  Note that the normal velocity component at the boundary ``u^\alpha_{I -
  \delta(\alpha) / 2}`` is now a degree of freedom, and we need to
  observe that the first normal derivative in the diffusion term of its
  corresponding momentum equation is zero before taking the limit as ``\epsilon
  \to 0``, to avoid dividing by zero.

It should now be clear from the above cases which components of the discrete
velocity and pressure are unknown degrees of freedom, and which components are
prescribed or obtained otherwise. The *unknown* degrees of freedom are stored
in the vectors ``u_h = (u^1_h, \dots, u^d_h)`` and ``p_h`` using the
column-major convention. Note that the ``d`` discrete velocity fields ``u^1_h,
\dots u^d_h`` may have different numbers of elements.

!!! note "Storage convention"
    We use the column-major convention (Julia, MATLAB, Fortran), and not the
    row-major convention (Python, C). Thus the ``x^1``-index ``i`` will vary for
    one whole cycle in the vectors before the
    ``x^2``-index ``j``, ``x^3`` index ``k``, and component-index ``\alpha``
    are incremented, e.g. ``u_h = (u^1_{(1, 1, 1)},
    u^1_{(2, 1, 1)}, \dots u^3_{(N_{u^3}(1), N_{u^3}(2), N_{u^3}(3))})`` in 3D.


## Fourth order accurate discretization

The above discretization is second order accurate.
A fourth order accurate discretization can be obtained by judiciously combining
the second order discretization with itself on a grid with three times larger
cells in each dimension [Verstappen2003](@cite) [Sanderse2014](@cite). The
coarse discretization is identical, but the mass equation is derived for the
three times coarser control volume

```math
\Omega^3_I =
\bigcup_{\alpha = 1}^d \Omega_{I - \delta(\alpha)} \cup
\Omega_I \cup \Omega_{I + \delta(\alpha)},
```
while the momentum equation is
derived for its shifted variant ``\Omega^3_{I + \delta(\alpha) / 2}``.
The resulting fourth order accurate equations are given by

```math
\sum_{\alpha = 1}^d
(\partial_\alpha u^\alpha)_I
-
\frac{| \Omega^3_I |}{3^{2 + d} | \Omega_I |}
\sum_{\alpha = 1}^d
(\partial^3_\alpha u^\alpha)_I
= 0
```

and

```math
\begin{split}
    \frac{\mathrm{d} }{\mathrm{d} t} u^\alpha_{I + \delta(\alpha) / 2} =
    - & \sum_{\beta = 1}^d
    (\partial_\beta (u^\alpha u^\beta))_{I + \delta(\alpha) / 2} \\
    + & \nu \sum_{\beta = 1}^d
    (\partial_\beta \partial_\beta u^\alpha)_{I + \delta(\alpha) / 2} \\
    + & f^\alpha(x_{I + \delta(\alpha) / 2}, t)
    - (\partial_\alpha p)_{I + \delta(\alpha) / 2}, \\
    + & \text{fourth order}
\end{split}
```

where

```math
(\partial^3_\alpha \varphi)_I =
\frac{\varphi_{I + 3 \delta(\alpha) / 2} -
\varphi_{I - 3 \delta(\alpha) / 2}}{\Delta^\alpha_{I(\alpha) - 1} +
\Delta^\alpha_{I(\alpha)} + \Delta^\alpha_{I(\alpha) + 1}}.
```

## Matrix representation

We can write the mass and momentum equations in matrix form. We will use the
same matrix notation for the second- and fourth order accurate discretizations.
The discrete mass equation becomes

```math
M u_h + y_M = 0,
```

where ``M`` is the discrete divergence operator and ``y_M`` contains the
boundary value contributions of the velocity to the divergence field.

The discrete momentum equations become

```math
\begin{split}
    \frac{\mathrm{d} u_h}{\mathrm{d} t} & = -C(u_h) + \nu (D u_h +
    y_D) + f_h - (G p_h + y_G) \\
    & = F(u_h) - (G p_h + y_G),
\end{split}
```

where ``C`` is she convection operator (including boundary contributions),
``D`` is the diffusion operator, ``y_D`` is boundary contribution to the
diffusion term,
``G = W_u^{-1} M^\mathsf{T} W`` is the pressure gradient
operator,
``y_G`` contains the boundary contribution of the pressure to the
pressure gradient (only non-zero for pressure boundary conditions),
``W_u`` is a diagonal matrix containing the velocity volume sizes
``| \Omega_{I + \delta(\alpha) / 2} |``, and ``W`` is a diagonal matrix
containing the reference volume sizes ``| \Omega_I |``.
The term ``F`` refers to all the forces except for the pressure gradient.

!!! note "Volume normalization"
    
    All the operators have been divided by the velocity volume sizes.
    As a result, the operators have the same units as their
    continuous counterparts.


## Discrete pressure Poisson equation

Instead of directly discretizing the continuous pressure Poisson equation, we
will rededuce it in the *discrete* setting, thus aiming to preserve the
discrete divergence freeness instead of the continuous one. Applying the
discrete divergence operator ``M`` to the discrete momentum equations yields
the discrete pressure Poisson equation

```math
L p_h = W M (F(u_h) - y_G) + W \frac{\mathrm{d} y_M}{\mathrm{d} t},
```

where ``L = W M G = W M W_u^{-1} M^\mathsf{T} W`` is a discrete Laplace
operator. It is positive symmetric.

!!! note "Unsteady Dirichlet boundary conditions"

    If the equations are prescribed with unsteady Dirichlet boundary
    conditions, for example an inflow that varies with time, the term
    ``\frac{\mathrm{d} y_M}{\mathrm{d} t}`` will be non-zero. If this term is
    not known exactly, for example if the next value of the inflow is unknown at
    the time of the current value, it must be computed using past values of
    of the velocity inflow only, for example ``\frac{\mathrm{d} y_M}{\mathrm{d}
    t} \approx (y_M(t) - y_M(t - \Delta t)) / \Delta t`` for some ``\Delta t``.

!!! note "Uniqueness of pressure field"

    Unless pressure boundary conditions are present, the pressure is only
    determined up to a constant, as ``L`` will have an eigenvalue of zero.
    Since only the gradient of the pressure appears in the equations, we can
    set the unknown constant to zero without affecting the velocity field.


!!! note "Pressure projection"

    The pressure field ``p_h`` can be seen as a Lagrange multiplier enforcing
    the constraint of discrete divergence freeness. It is also possible to
    write the momentum equations without the pressure by explicitly solving the
    discrete Poisson equation:

    ```math
    p_h = L^{-1} W M (F(u_h) - y_G) + L^{-1} W \frac{\mathrm{d} y_M}{\mathrm{d} t}.
    ```

    The momentum equations then become

    ```math
    \frac{\mathrm{d} u_h}{\mathrm{d} t} = (I - G L^{-1} W M)
    (F(u_h) - y_G) - G L^{-1} W \frac{\mathrm{d} y_M}{\mathrm{d} t}.
    ```

    The matrix ``(I - G L^{-1} W M)`` is a projector onto the space
    of discretely divergence free velocities. However, using this formulation
    would require an efficient way to perform the projection without assembling
    the operator matrix ``L^{-1}``, which would be very costly.

## Discrete output quantities

### Kinetic energy

The local kinetic energy is defined by ``k = \frac{1}{2} \| u \|_2^2 =
\frac{1}{2} \sum_{\alpha = 1}^d u^\alpha u^\alpha``. On the staggered grid
however, the different velocity components are not located at the same point.
We will therefore interpolate the velocity to the pressure point before summing
the squares.

### Vorticity

In 2D, the vorticity is a scalar. Integrating the vorticity ``\omega =
-\frac{\partial u^1}{\partial x^2} + \frac{\partial u^2}{\partial x^1}`` over
the vorticity volume ``\Omega_{I + \delta(1) / 2 + \delta(2) / 2}`` gives

```math
\begin{split}
\int_{\Omega_{I + \delta(1) / 2 + \delta(2) / 2}} \omega \, \mathrm{d} \Omega
= & - \left(
\int_{\Gamma^2_{I + \delta(1) / 2 + \delta(2)}} u^1 \, \mathrm{d} \Gamma
- \int_{\Gamma^2_{I + \delta(1) / 2}} u^1 \, \mathrm{d} \Gamma
\right) \\
& + \left(
\int_{\Gamma^1_{I + \delta(1) + \delta(2) / 2}} u^2 \, \mathrm{d} \Gamma
- \int_{\Gamma^1_{I + \delta(2) / 2}} u^2 \, \mathrm{d} \Gamma
\right)
\end{split}.
```

Using quadrature, and dividing by the vorticity volume
``| \Omega_{I + \delta(1) / 2 + \delta(2) / 2} |``,
the discrete vorticity in the corner is given by

```math
\omega_{I + \delta(1) / 2 + \delta(2) / 2} =
- \frac{u^1_{I + \delta(1) / 2 + \delta(2)} -
u^1_{I + \delta(1) / 2}}{\Delta^2_{I(2) + 1 / 2}}
+ \frac{u^2_{I + \delta(1) + \delta(2) / 2} -
u^2_{I + \delta(2) / 2}}{\Delta^1_{I(1) + 1 / 2}}.
```

The 3D vorticity is a vector field ``(\omega^1, \omega^2, \omega^3)``.
Noting ``\alpha^+ = \operatorname{mod}_3(\alpha + 1)`` and
``\alpha^- = \operatorname{mod}_3(\alpha - 1)``, the vorticity is obtained
through

```math
\begin{split}
\int_{\Omega_{I + \delta(\alpha^+) / 2 + \delta(\alpha^-) / 2}} \omega \, \mathrm{d} \Omega
= & - \left(
\int_{\Gamma^{\alpha^-}_{I + \delta(\alpha^+) / 2 + \delta(\alpha^-)}} u^{\alpha^+} \, \mathrm{d} \Gamma
- \int_{\Gamma^{\alpha^-}_{I + \delta(\alpha^+) / 2}} u^{\alpha^+} \, \mathrm{d} \Gamma
\right) \\
& + \left(
\int_{\Gamma^{\alpha^+}_{I + \delta(\alpha^+) + \delta(\alpha^-) / 2}} u^{\alpha^-} \, \mathrm{d} \Gamma
- \int_{\Gamma^{\alpha^+}_{I + \delta(\alpha^-) / 2}} u^{\alpha^-} \, \mathrm{d} \Gamma
\right)
\end{split}.
```

Using quadrature, and dividing by the vorticity volume
``| \Omega_{I + \delta(\alpha^+) / 2 + \delta(\alpha^-) / 2} |``,
the discrete vorticity around the ``\alpha``-edge is given by

```math
\omega_{I + \delta(\alpha^+) / 2 + \delta(\alpha^-) / 2} =
- \frac{u^{\alpha^+}_{I + \delta(\alpha^+) / 2 + \delta(\alpha^-)} -
u^{\alpha^+}_{I + \delta(\alpha^+) / 2}}{\Delta^{\alpha^-}_{I(\alpha^-) + 1 / 2}}
+ \frac{u^{\alpha^-}_{I + \delta(\alpha^+) + \delta(\alpha^-) / 2} -
u^{\alpha^-}_{I + \delta(\alpha^-) / 2}}{\Delta^{\alpha^+}_{I(\alpha^+) + 1 /
2}}.
```

## Stream function

In 2D, the stream function is defined at the corners with the vorticity.
Integrating the stream function Poisson equation over the vorticity volume
yields

```math
\begin{split}
- \int_{\Omega_{I + \delta(1) / 2 + \delta(2) / 2}} \omega \, \mathrm{d} \Omega
& = \int_{\Omega_{I + \delta(1) / 2 + \delta(2) / 2}} \nabla^2 \psi \,
\mathrm{d} \Omega \\
& = \int_{\Gamma^1_{I + \delta(1) + \delta(2) / 2}} \frac{\partial \psi}{\partial x^1}
\, \mathrm{d} \Gamma
- \int_{\Gamma^1_{I + \delta(2) / 2}} \frac{\partial \psi}{\partial x^1}
\, \mathrm{d} \Gamma \\
& + \int_{\Gamma^2_{I + \delta(1) / 2 + \delta(2)}} \frac{\partial \psi}{\partial x^2}
\, \mathrm{d} \Gamma
- \int_{\Gamma^2_{I + \delta(1) / 2}} \frac{\partial \psi}{\partial x^2}
\, \mathrm{d} \Gamma.
\end{split}
```

Replacing the integrals with the mid-point quadrature rule and the spatial
derivatives with central finite differences yields the discrete Poisson
equation for the stream function at the vorticity point:

```math
\begin{split}
\left| \Gamma^1_{I + \delta(1) / 2 + \delta(2) / 2} \right|
\left(
  \frac{\psi_{I + 3 / 2 \delta(1) + \delta(2) / 2} - \psi_{I + \delta(1) / 2 + \delta(2) / 2}}{x^1_{I(1) + 3 / 2} - x^1_{I(1) + 1 /2}}
- \frac{\psi_{I + \delta(1) / 2 + \delta(2) / 2} - \psi_{I - \delta(1) / 2 + \delta(2) / 2}}{x^1_{I(1) + 1 / 2} - x^1_{I(1) - 1 / 2}}
\right) & + \\
\left| \Gamma^2_{I + \delta(1) / 2 + \delta(2) / 2} \right|
\left(
\frac{\psi_{I + \delta(1) / 2 + 3 / 2 \delta(2)} - \psi_{I + \delta(1) / 2 + \delta(2) / 2}}{x^2_{I(1) + 3 / 2} - x^2_{I(1) + 1 / 2}}
-\frac{\psi_{I + \delta(1) / 2 + \delta(2) / 2} - \psi_{I + \delta(1) / 2 - \delta(2) / 2}}{x^2_{I(2) + 1 / 2} - x^2_{I(2) - 1 / 2}}
\right) & = \\
\left| \Omega_{I + \delta(1) / 2 + \delta(2) / 2} \right|
\omega_{I + \delta(1) / 2 + \delta(2) / 2} &
\end{split}
```
