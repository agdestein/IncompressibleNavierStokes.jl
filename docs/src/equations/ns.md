# Incompressible Navier-Stokes equations

Let ``d \in \{2, 3\}`` denote the spatial dimension, and ``\Omega \subset
\mathbb{R}^d`` some spatial domain. The incompressible Navier-Stokes equations
on ``\Omega`` are comprised of a mass equation and ``d`` momentum equations. In
conservative form, they are given by

```math
\begin{align*}
\nabla \cdot u & = 0, \\
\frac{\partial u}{\partial t} + \nabla \cdot (u u^\mathsf{T}) & = -\nabla p +
\nu \nabla^2 u + f.
\end{align*}
```

where ``u = (u^1, \dots, u^d)`` is the velocity field, ``p`` is the
pressure, ``\nu`` is the kinematic viscosity, and ``f = (f^1, \dots, f^d)`` is
the body force per unit of volume. In scalar notation, the equations can be
written as

```math
\begin{align*}
\sum_{\beta = 1}^d \frac{\partial u^\beta}{\partial x^\beta} & = 0, \\
\frac{\partial u^\alpha}{\partial t} + \sum_{\beta = 1}^d
\frac{\partial}{\partial x^\beta} (u^\alpha u^\beta) & = -\frac{\partial
p}{\partial x^\alpha} + \nu \sum_{\beta = 1}^d \frac{\partial^2 u^\alpha}{\partial
(x^\beta)^2} + f^\alpha.
\end{align*}
```


## Integral form

When discretizing the Navier-Stokes equations it can be useful to integrate the
equations over an arbitrary control volume ``\mathcal{O}``. Its boundary will
be denoted ``\partial \mathcal{O}``, with normal ``n`` and surface element
``\mathrm{d} \Gamma``.

The mass equation in integral form is given by

```math
\int_{\partial \mathcal{O}} u \cdot n \, \mathrm{d} \Gamma = 0,
```

where we have used the divergence theorem to convert the volume integral to a
surface integral. Similarly, the momentum equations take the form

```math
\frac{\partial }{\partial t} \int_\mathcal{O} u \, \mathrm{d} \Omega
= \int_{\partial \mathcal{O}} \left( - u u^\mathsf{T} - P + \nu S \right) \cdot n \,
\mathrm{d} \Gamma + \int_\mathcal{O} f \mathrm{d} \Omega
```

where ``P = p \mathrm{I}`` is the hydrostatic stress tensor
and ``S = \nabla u + (\nabla u)^\mathsf{T}`` is the strain-rate tensor. Here we
have once again used the divergence theorem.

!!! note "Strain-rate tensor"
    The second term in the strain rate tensor is optional, as

    ```math
    \int_{\partial \mathcal{O}} (\nabla u)^\mathsf{T} \cdot n \, \mathrm{d} \Gamma = 0
    ```

    due to the divergence freeness of ``u`` (mass equation).


## Boundary conditions

Consider a part ``\Gamma`` of the total boundary ``\partial \Omega``, with
normal ``n``. We allow for the following types of boundary conditions on
``\Gamma``:

- Periodic boundary conditions: ``u(x) = u(x + \tau)`` and ``p(x) = p(x + \tau)`` for ``x \in
  \Gamma``, where ``\tau`` is a constant translation to another part of the
  boundary ``\partial \Omega``. This usually requires ``\Gamma`` and its
  periodic counterpart ``\Gamma + \tau`` to be flat and rectangular (including
  in this software suite).
- Dirichlet boundary conditions: ``u = u_\text{BC}`` on ``\Gamma``. A common
  situation here is that of a sticky wall, with "no slip boundary conditions.
  Then ``u_\text{BC} = 0``.
- Symmetric boundary conditions: Same velocity field at each side. This implies
  zero Dirichlet conditions for the normal component (``n \cdot u = 0``), and
  zero Neumann conditions for the parallel component: ``n \cdot \nabla (u - (n
  \cdot u) n) = 0``.
- Pressure boundary conditions: The pressure is prescribed (``p =
  p_\text{BC}``), while the velocity has zero Neumann conditions:
  ``n \cdot \nabla u = 0``.


## Pressure equation

Taking the divergence of the momemtum equations yields a Poisson
equation for the pressure:

```math
- \nabla^2 p = \nabla \cdot \left( \nabla \cdot (u u^\mathsf{T}) \right) -
\nabla \cdot f
```

In scalar notation, this becomes

```math
- \sum_{\alpha = 1}^d \frac{\partial^2}{\partial (x^\alpha)^2} p = \sum_{\alpha
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

The stream function ``\psi`` is a field (scalar in 2D, vector in 3D) defined
such that

```math
u = \nabla \times \psi.
```

It is related to the vorticity as

```math
\nabla^2 \psi = \nabla \times \omega.
```

### Kinetic energy

The local and total kinetic energy are defined by ``k = \frac{1}{2} \| u
\|_2^2`` and ``K = \frac{1}{2} \| u \|_{L^2(\Omega)}^2 = \int_\Omega k \,
\mathrm{d} \Omega``.

### Reynolds number

The Reynolds number is the inverse of the viscosity: ``Re =
\frac{1}{\nu}``.
