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

The equations are stated here in dimensionless form with a reference length
and velocity of unity, in which case the viscosity is the inverse of the
Reynolds number: ``\nu = 1 / Re``. In the code, the viscosity is passed
directly, e.g. `params = (; viscosity = 1e-3)`.

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
integral form, so that all terms have the same units as their differential
counterparts.

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

See [Problem setup](setup.md) for how to prescribe boundary conditions in the
code.

## Pressure equation

Taking the divergence of the momentum equations yields a Poisson
equation for the pressure:

```math
- \nabla^2 p = \nabla \cdot \left( \nabla \cdot (u u^\mathsf{T}) \right) -
\nabla \cdot f
```

Note the absence of time derivatives in the pressure equation. While the
velocity field evolves in time, the pressure only changes such that the
velocity stays divergence free.

If there are no pressure boundary conditions, the pressure is only unique up to
a constant. Since only the gradient of the pressure appears in the equations,
this constant can be set to zero without affecting the velocity field.

## Other quantities of interest

### Kinetic energy

The local and total kinetic energy are defined by ``k = \frac{1}{2} \| u
\|_2^2`` and ``K = \frac{1}{2} \| u \|_{L^2(\Omega)}^2 = \int_\Omega k \,
\mathrm{d} \Omega``. In the absence of viscosity, boundaries, and body forces,
the total kinetic energy is conserved. The discretization used in this package
preserves this property (see [Spatial and temporal
discretization](discretization.md)).

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

Note that the 2D vorticity is equal to the ``x^3``-component of the 3D
vorticity.
