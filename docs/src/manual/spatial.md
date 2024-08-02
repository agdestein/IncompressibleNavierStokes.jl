# Spatial discretization

the ``d`` spatial dimensions are indexed by ``\alpha \in \{1, \dots, d\}``. The
``\alpha``-th unit vector is denoted ``e_\alpha = (e_{\alpha \beta})_{\beta =
1}^d``, where the Kronecker symbol ``e_{\alpha \beta}`` is ``1`` if ``\alpha =
\beta`` and ``0`` otherwise. We note ``h_\alpha = e_\alpha / 2``. The Cartesian
index ``I = (I_1, \dots, I_d)`` is used to avoid repeating terms and equations
``d`` times, where ``I_\alpha`` is a scalar index (typically one of ``i``,
``j``, and ``k`` in common notation). This notation is dimension-agnostic, since
we can write ``u_I`` instead of ``u_{i j}`` in 2D or ``u_{i j k}`` in 3D. In our
Julia implementation of the solver we use the same Cartesian notation
(`u[I]` instead of `u[i, j]` or `u[i, j, k]`).

For the discretization scheme, we use a staggered Cartesian grid as proposed by
Harlow and Welch [Harlow1965](@cite). Consider a rectangular domain
``\Omega = \prod_{\alpha = 1}^d [a_\alpha, b_\alpha]``, where ``a_\alpha <
b_\alpha`` are the domain boundaries and ``\prod`` is a Cartesian product. Let
``\Omega = \bigcup_{I \in \mathcal{I}} \Omega_I`` be a partitioning of ``\Omega``,
where ``\mathcal{I} = \prod_{\alpha = 1}^d \{ \frac{1}{2}, 2 - \frac{1}{2},
\dots, N_\alpha - \frac{1}{2} \}`` are volume center indices, ``N = (N_1, \dots,
N_d) \in \mathbb{N}^d`` are the number of volumes in each dimension,
``\Omega_I = \prod_{\alpha = 1}^d \Delta^\alpha_{I_\alpha}`` is a finite
volume, ``\Gamma^\alpha_I = \Omega_{I - h_\alpha} \cap \Omega_{I + h_\alpha} =
\prod_{\beta \neq \alpha} \Delta^\beta_{I_\beta}`` is a volume face,
``\Delta^\alpha_i = \left[ x^\alpha_{i - \frac{1}{2}}, x^\alpha_{i + \frac{1}{2}}
\right]`` is a volume edge, ``x^\alpha_0, \dots, x^\alpha_{N_\alpha}`` are volume
boundary coordinates, and ``x^\alpha_i = \frac{1}{2} \left(x^\alpha_{i -
\frac{1}{2}} + x^\alpha_{i + \frac{1}{2}}\right)`` for ``i \in \{ 1 / 2, \dots,
N_\alpha - 1 / 2\}`` are volume center coordinates. We also define the operator
``\delta_\alpha`` which maps a discrete scalar field ``\varphi = (\varphi_I)_I`` to

```math
(\delta_\alpha \varphi)_I = \frac{\varphi_{I + h_\alpha} - \varphi_{I -
h_\alpha}}{| \Delta^\alpha_{I_\alpha} |}.
```

It can be interpreted as a discrete equivalent of the continuous operator
``\frac{\partial}{\partial x^\alpha}``. All the above definitions are extended to
be valid in volume centers ``I \in \mathcal{I}``, volume faces
``I \in \mathcal{I} + h_\alpha``,
or volume corners ``I \in \mathcal{I} + \sum_{\alpha = 1}^d h_\alpha``.
The discretization is illustrated below:

![Grid](../public/grid.png)

## Finite volume discretization of the Navier-Stokes equations

We now define the unknown degrees of freedom. The average pressure in
``\Omega_I``, ``I \in \mathcal{I}`` is approximated by the quantity ``p_I(t)``. The
average ``\alpha``-velocity on the face ``\Gamma^\alpha_I``, ``I \in \mathcal{I} +
h_\alpha`` is approximated by the quantity ``u^\alpha_I(t)``. Note how the pressure
``p`` and the ``d`` velocity fields ``u^\alpha`` are each defined in their own
canonical positions ``x_I`` and ``x_{I + h_\alpha}`` for ``I \in \mathcal{I}``.
In the following, we derive equations for these unknowns.

Using the pressure control volume ``\mathcal{O} = \Omega_I`` with ``I \in
\mathcal{I}`` in the mass integral constraint and
approximating the face integrals with the mid-point quadrature rule
``\int_{\Gamma_I} u \, \mathrm{d} \Gamma \approx | \Gamma_I | u_I`` results in the
discrete divergence-free constraint

```math
\sum_{\alpha = 1}^d (\delta_\alpha u^\alpha)_I = 0.
```

Note how dividing by the volume size results in a discrete equation resembling
the continuous one (since ``| \Omega_I | = | \Gamma^\alpha_I | |
\Delta^\alpha_{I_\alpha} |``).

Similarly, choosing an ``\alpha``-velocity control volume ``\mathcal{O} =
\Omega_{I}`` with ``I \in \mathcal{I} + h_\alpha`` in the integral momentum
equation, approximating the volume- and face integrals using
the mid-point quadrature rule, and replacing remaining spatial derivatives in
the diffusive term with a finite difference approximation gives the discrete
momentum equations

```math
\frac{\mathrm{d}}{\mathrm{d} t} u^\alpha_{I} =
- \sum_{\beta = 1}^d
(\delta_\beta (u^\alpha u^\beta))_{I}
+ \nu \sum_{\beta = 1}^d
(\delta_\beta \delta_\beta u^\alpha)_{I}
+ f^\alpha(x_{I})
- (\delta_\alpha p)_{I}.
```

where we made the assumption that ``f`` is constant in time for simplicity.
The outer discrete derivative in ``(\delta_\beta \delta_\beta u^\alpha)_{I}`` is
required at the position ``I``, which means that the inner derivative is evaluated
as ``(\delta_\beta u^\alpha)_{I + h_\beta}`` and ``(\delta_\beta u^\alpha)_{I -
h_\beta}``, thus requiring ``u^\alpha_{I - 2 h_\beta}``, ``u^\alpha_{I}``, and
``u^\alpha_{I + 2 h_\beta}``, which are all in their canonical positions. The two
velocity components in the convective term ``u^\alpha u^\beta`` are required at
the positions ``I - h_\beta`` and ``I + h_\beta``, which are outside the canonical
positions. Their value at the required position is obtained using averaging with
weights ``1 / 2`` for the ``\alpha``-component and with linear interpolation for the
``\beta``-component. This preserves the skew-symmetry of the convection operator,
such that energy is conserved (in the convective term) [Verstappen2003](@cite).

## Boundary conditions

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
\bigcup_{\alpha = 1}^d \Omega_{I - e_\alpha} \cup
\Omega_I \cup \Omega_{I + e_\alpha},
```
while the momentum equation is
derived for its shifted variant ``\Omega^3_{I + h_\alpha}``.
The resulting fourth order accurate equations are given by

```math
\sum_{\alpha = 1}^d
(\delta_\alpha u^\alpha)_I
-
\frac{| \Omega^3_I |}{3^{2 + d} | \Omega_I |}
\sum_{\alpha = 1}^d
(\delta^3_\alpha u^\alpha)_I
= 0
```

and

```math
\frac{\mathrm{d}}{\mathrm{d} t} u^\alpha_{I} =
- \sum_{\beta = 1}^d
(\delta_\beta (u^\alpha u^\beta))_{I}
+ \nu \sum_{\beta = 1}^d
(\delta_\beta \delta_\beta u^\alpha)_{I}
+ f^\alpha(x_{I})
- (\delta_\alpha p)_{I}
+ \text{fourth order},
```

where

```math
(\delta^3_\alpha \varphi)_I =
\frac{\varphi_{I + 3 h_\alpha} -
\varphi_{I - 3 h_\alpha}}{\Delta^\alpha_{I_\alpha - 1} +
\Delta^\alpha_{I_\alpha} + \Delta^\alpha_{I_\alpha + 1}}.
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

In 2D, the vorticity is a scalar. We define it as

```math
\omega = -\delta^2 u^1 + \delta^1 u^2.
```

The 3D vorticity is a vector field ``(\omega^1, \omega^2, \omega^3)``.
Noting ``\alpha^+ = \operatorname{mod}_3(\alpha + 1)`` and
``\alpha^- = \operatorname{mod}_3(\alpha - 1)``, the vorticity is defined as
through

```math
\omega^\alpha = - \delta^{\alpha^-} u^{\alpha^+} +
\delta^{\alpha^+} u^{\alpha^-}.
```

## Stream function

In 2D, the stream function is defined at the corners with the vorticity.
Integrating the stream function Poisson equation over the vorticity volume
yields

```math
\begin{split}
- \int_{\Omega_{I + h_1 + h_2}} \omega \, \mathrm{d} \Omega
& = \int_{\Omega_{I + h_1 + h_2}} \nabla^2 \psi \,
\mathrm{d} \Omega \\
& = \int_{\Gamma^1_{I + e_1 + h_2}} \frac{\partial \psi}{\partial x^1}
\, \mathrm{d} \Gamma
- \int_{\Gamma^1_{I + h_2}} \frac{\partial \psi}{\partial x^1}
\, \mathrm{d} \Gamma \\
& + \int_{\Gamma^2_{I + h_1 + e_2}} \frac{\partial \psi}{\partial x^2}
\, \mathrm{d} \Gamma
- \int_{\Gamma^2_{I + h_1}} \frac{\partial \psi}{\partial x^2}
\, \mathrm{d} \Gamma.
\end{split}
```

Replacing the integrals with the mid-point quadrature rule and the spatial
derivatives with central finite differences yields the discrete Poisson
equation for the stream function at the vorticity point:

```math
\begin{split}
\left| \Gamma^1_{I + h_1 + h_2} \right|
\left(
  \frac{\psi_{I + 3 / h_1 + h_2} - \psi_{I + h_1 + h_2}}{x^1_{I_1 + 3 / 2} - x^1_{I_1 + 1 /2}}
- \frac{\psi_{I + h_1 + h_2} - \psi_{I - h_1 + h_2}}{x^1_{I_1 + 1 / 2} - x^1_{I_1 - 1 / 2}}
\right) & + \\
\left| \Gamma^2_{I + h_1 + h_2} \right|
\left(
\frac{\psi_{I + h_1 + 3 h_2} - \psi_{I + h_1 + h_2}}{x^2_{I_1 + 3 / 2} - x^2_{I_1 + 1 / 2}}
-\frac{\psi_{I + h_1 + h_2} - \psi_{I + h_1 - h_2}}{x^2_{I_2 + 1 / 2} - x^2_{I_2 - 1 / 2}}
\right) & = \\
\left| \Omega_{I + h_1 + h_2} \right|
\omega_{I + h_1 + h_2} &
\end{split}
```
