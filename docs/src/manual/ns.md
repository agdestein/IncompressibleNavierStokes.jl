# Incompressible Navier-Stokes equations

The incompressible Navier-Stokes equations describe conservation of mass and
conservation of momentum, which can be written as a divergence-free constraint
and an evolution equation:

```math
\begin{aligned}
    \nabla \cdot u & = 0, \\
    \frac{\partial u}{\partial t} + \nabla \cdot (u u^\mathsf{T})
    & = -\nabla p + \nu \nabla^2 u + f,
\end{aligned}
```

where ``\Omega \subset \mathbb{R}^d`` is the domain, ``d \in \{2, 3\}`` is the
spatial dimension, ``u = (u^1, \dots, u^d)`` is the velocity field, ``p`` is the
pressure, ``\nu`` is the kinematic viscosity, and ``f = (f^1, \dots, f^d)`` is
the body force per unit of volume. The velocity, pressure, and body force are
functions of the spatial coordinate ``x = (x^1, \dots, x^d)`` and time ``t``.
We assume that ``\Omega`` is a rectangular domain.

## Integral form

The integral form of the Navier-Stokes equations is used as starting point to
develop a spatial discretization:

```math
\begin{aligned}
    \frac{1}{|\mathcal{O}|} \int_{\partial \mathcal{O}}
    u \cdot n \, \mathrm{d} \Gamma & = 0, \\
    \frac{\mathrm{d} }{\mathrm{d} t} \frac{1}{|\mathcal{O}|}
    \int_\mathcal{O} u \, \mathrm{d} \Omega
    & = \frac{1}{|\mathcal{O}|} \int_{\partial \mathcal{O}}
    \left( - u u^\mathsf{T} - p I + \nu \nabla u \right) \cdot n \,
    \mathrm{d} \Gamma +
    \frac{1}{|\mathcal{O}|}\int_\mathcal{O} f \mathrm{d} \Omega,
\end{aligned}
```

where ``\mathcal{O} \subset \Omega`` is an arbitrary control volume with boundary
``\partial \mathcal{O}``, normal ``n``, surface element ``\mathrm{d} \Gamma``, and
volume size ``|\mathcal{O}|``. We have divided by the control volume sizes in the
integral form.

## Boundary conditions

The boundary conditions on a part of the boundary
``\Gamma \subset \partial \Omega`` are one or more of the following:

- Dirichlet: ``u = u_\text{BC}`` on ``\Gamma`` for some ``u_\text{BC}``;
- Neumann: ``\nabla u \cdot n = 0`` on ``\Gamma``;
- Periodic: ``u(x) = u(x + \tau)`` and ``p(x) = p(x + \tau)``
    for ``x \in \Gamma``, where
    ``\Gamma + \tau \subset \partial \Omega``
    is another part of the boundary and
    ``\tau`` is a translation vector;
- Stress free: ``\sigma \cdot n = 0`` on ``\Gamma``,
    where ``\sigma = \left(- p \mathrm{I} + 2 \nu S \right)``.

## Pressure equation

Taking the divergence of the momemtum equations yields a Poisson
equation for the pressure:

```math
- \nabla^2 p = \nabla \cdot \left( \nabla \cdot (u u^\mathsf{T}) \right) -
\nabla \cdot f
```

In scalar notation, this becomes

```math
- \sum_{\alpha = 1}^d \frac{\partial^2}{\partial x^\alpha \partial x^\alpha} p = \sum_{\alpha
= 1}^d \sum_{\beta = 1}^d \frac{\partial^2 }{\partial x^\alpha \partial
x^\beta} (u^\alpha u^\beta) - \sum_{\alpha = 1}^d \frac{\partial
f^\alpha}{\partial x^\alpha}.
```

Note the absence of time derivatives in the pressure equation. While the
velocity field evolves in time, the pressure only changes such that the
velocity stays divergence free.

If there are no pressure boundary conditions, the pressure is only unique up to
a constant. We set this constant to ``1``.


## Other quantities of interest

### Reynolds number

The Reynolds number is the inverse of the viscosity: ``Re = \frac{1}{\nu}``. It
is the only flow parameter governing the incompressible Navier-Stokes
equations.

### Kinetic energy

The local and total kinetic energy are defined by ``k = \frac{1}{2} \| u
\|_2^2`` and ``K = \frac{1}{2} \| u \|_{L^2(\Omega)}^2 = \int_\Omega k \,
\mathrm{d} \Omega``.

### Vorticity

The vorticity is defined as ``\omega = \nabla \times u``.

In 2D, it is a scalar field given by

```math
\omega = -\frac{\partial u^1}{\partial x^2} + \frac{\partial u^2}{\partial
x^1}.
```

In 3D, it is a vector field given by

```math
\omega = \begin{pmatrix}
    - \frac{\partial u^2}{\partial x^3} + \frac{\partial u^3}{\partial x^2} \\
    - \frac{\partial u^3}{\partial x^1} + \frac{\partial u^1}{\partial x^3}  \\
    - \frac{\partial u^1}{\partial x^2} + \frac{\partial u^2}{\partial x^1}
\end{pmatrix}.
```

Note that the 2D vorticity is equal
to the ``x^3``-component of the 3D vorticity.

### Stream function

In 2D, the stream function ``\psi`` is a scalar field such that

```math
u^1 = \frac{\partial \psi}{\partial x^2}, \quad
u^2 = -\frac{\partial \psi}{\partial x^1}.
```

It can be found by solving

```math
\nabla^2 \psi = - \omega.
```

In 3D, the stream function is a vector field such that

```math
u = \nabla \times \psi.
```

It can be found by solving

```math
\nabla^2 \psi = \nabla \times \omega.
```
