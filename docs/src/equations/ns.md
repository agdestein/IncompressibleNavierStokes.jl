# Incompressible Navier-Stokes equations

The incompressible Navier-Stokes equations are comprised of a mass equation and
two or three momentum equations. In conservative form, they are given by

```math
\begin{align*}
\nabla \cdot V & = 0, \\
\frac{\mathrm{d} V}{\mathrm{d} t} + \nabla \cdot (V V^\mathsf{T}) & = -\nabla p +
\nu \nabla^2 V + f.
\end{align*}
```

where ``V = (u, v)`` or ``V = (u, v, w)`` is the velocity field, ``p`` is the
pressure, ``\nu`` is the kinematic viscosity, and ``f = (f_u, f_v)`` or ``f =
(f_u, f_v, f_w)`` is the body force. In 2D, the equations become

```math
\begin{split}
    \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} & = 0, \\
    \frac{\partial u}{\partial t} + \frac{\partial (u u)}{\partial x} +
    \frac{\partial (v u)}{\partial y} & = - \frac{\partial p}{\partial x} +
    \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2
    u}{\partial y^2} \right) + f_u, \\
    \frac{\partial v}{\partial t} + \frac{\partial (u v)}{\partial x} +
    \frac{\partial (v v)}{\partial y} & = - \frac{\partial p}{\partial y} +
    \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2
    v}{\partial y^2} \right) + f_v.
\end{split}
```

In 3D, the equations become

```math
\begin{split}
    \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} +
    \frac{\partial w}{\partial z} & = 0, \\
    \frac{\partial u}{\partial t} + \frac{\partial (u u)}{\partial x} +
    \frac{\partial (v u)}{\partial y} + \frac{\partial (w u)}{\partial z} & = -
    \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial
    x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial
    z^2} \right) + f_u, \\
    \frac{\partial v}{\partial t} + \frac{\partial (u v)}{\partial x} +
    \frac{\partial (v v)}{\partial y} + \frac{\partial (w v)}{\partial z} & = -
    \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial
    x^2} + \frac{\partial^2 v}{\partial y^2} + \frac{\partial^2 v}{\partial
    z^2}  \right) + f_v, \\
    \frac{\partial w}{\partial t} + \frac{\partial (u w)}{\partial x} +
    \frac{\partial (v w)}{\partial y} + \frac{\partial (w w)}{\partial z} & = -
    \frac{\partial p}{\partial z} + \nu \left( \frac{\partial^2 w}{\partial
    x^2} + \frac{\partial^2 w}{\partial y^2} + \frac{\partial^2 w}{\partial
    z^2}  \right) + f_w. \\
\end{split}
```

## Boundary conditions

Because we will use a Cartesian grid for discretization, the fields ``V``,
``p``, and ``f`` are defined over the rectangular or prismatic domain ``\Omega
= [x_1, x_2] \times [y_1, y_2]`` or ``\Omega = [x_1, x_2] \times [y_1, y_2]
\times [z_1, z_2]``. Along each of the two or three dimensions, we allow for
the following boundary conditions (here illustrated on the first boundary of
the first dimension)

- Periodic: ``V(x = x_1) = V(x = x_2)`` and ``p(x = x_1) = p(x = x_2)``
- Dirichlet: ``V(x = x_1) = V_1``
- Pressure: ``p(x = x_1) = p_1``
- Symmetric (no movement through boundary ``x = x_1``, but free movement along
  it): ``u(x = x_1) = 0``

## Pressure equation

Taking the divergence of the two or tree momemtum equations yields a Poisson
equation for the pressure:

```math
- \nabla^2 p = \nabla \cdot \left( \nabla \cdot (V V^\mathsf{T}) \right) -
\nabla \cdot f
```

In 2D, this equation becomes

```math
- \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial
y^2}\right) p = \frac{\partial^2 (u u)}{\partial x^2} + 2 \frac{\partial^2 (u
v)}{\partial x \partial y} + \frac{\partial^2 (v v)}{\partial y^2} - \left( \frac{\partial f_u}{\partial x} + \frac{\partial f_v}{\partial y} \right).
```

In 3D, this equation becomes

```math
\begin{split}
    - \left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial
    y^2} + \frac{\partial^2 }{\partial z^2} \right) p
    & = \frac{\partial^2 (u u)}{\partial x^2}
    + 2 \frac{\partial^2 (u v)}{\partial x \partial y}
    + 2 \frac{\partial^2 (u w)}{\partial x \partial z} \\
    & + 2 \frac{\partial^2 (v u)}{\partial y \partial x}
    +   \frac{\partial^2 (v v)}{\partial y^2}
    + 2 \frac{\partial^2 (v w)}{\partial y \partial z} \\
    & + 2 \frac{\partial^2 (v u)}{\partial z \partial x}
    + 2 \frac{\partial^2 (w v)}{\partial z \partial y}
    +   \frac{\partial^2 (w w)}{\partial z^2} \\
    & - \left( \frac{\partial f_u}{\partial x} + \frac{\partial f_v}{\partial
    y} + \frac{\partial f_w}{\partial z} \right).
\end{split}
```

Note the absence of time derivatives in the pressure equation. While the
velocity field evolves in time, the pressure only changes such that the
velocity stays divergence free.

If there are no pressure boundary conditions, the pressure is only unique up to
a constant. We set this constant to ``1``.

## Other quantities of interest

### Vorticity

The vorticity is defined as ``\omega = \nabla \times V``.

In 2D, it is a scalar field given by

```math
\omega = -\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}.
```

In 3D, it is a vector field given by

```math
\omega = \begin{pmatrix}
    - \frac{\partial v}{\partial z} + \frac{\partial w}{\partial y} \\
    \phantom{+} \frac{\partial u}{\partial z} - \frac{\partial w}{\partial x} \\
    - \frac{\partial u}{\partial z} + \frac{\partial w}{\partial x}
\end{pmatrix}.
```

Note that the 2D vorticity is equal
to the ``z``-component of the 3D vorticity.

### Stream function

In 2D, the stream function ``\psi`` is a scalar field defined such that

```math
u = \frac{\partial \psi}{\partial y}, \quad v = \frac{\partial \psi}{\partial x}.
```

In 3D, the stream function ``\psi`` is a vector field defined such that

```math
V = \nabla \times \psi
```

It is related to the 3D vorticity as

```math
\nabla^2 \psi = \nabla \times \omega
```

### Kinetic energy

The kinetic energy is defined by ``e = \frac{1}{2} \| V \|^2``.

### Reynolds number

The Reynolds number is the inverse of the viscosity: ``Re =
\frac{1}{\nu}``.
