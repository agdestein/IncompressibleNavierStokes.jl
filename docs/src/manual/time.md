```@meta
CurrentModule = IncompressibleNavierStokes
```

# Time discretization

The spatially discretized Navier-Stokes equations form a differential-algebraic
system, with an ODE for the velocity

```math
\frac{\mathrm{d} u}{\mathrm{d} t} = F(u, t) - (G p + y_G)
```

subject to the algebraic constraint formed by the mass equation

```math
M u + y_M = 0.
```

In the end of the previous section, we differentiated the mass
equation in time to obtain a discrete pressure Poisson equation. This equation
includes the term ``\frac{\mathrm{d} y_M}{\mathrm{d} t}``, which is non-zero if
an unsteady flow of mass is added to the domain (Dirichlet boundary
conditions). This term ensures that the time-continuous discrete velocity field
``u(t)`` stays divergence free (conserves mass). However, if we directly
discretize this system in time, the mass preservation may actually not be
respected. For this, we will change the definition of the pressure such that
the time-discretized velocity field is divergence free at each time step and
each time sub-step (to be defined in the following).

Consider the interval ``[0, T]`` for some simulation time ``T``. We will divide
it into ``N`` sub-intervals ``[t^n, t^{n + 1}]`` for ``n = 0, \dots, N - 1``,
with ``t^0 = 0``, ``t^N = T``, and increment ``\Delta t^n = t^{n + 1} - t^n``.
We define ``u^n \approx u(t^n)`` as an approximation to the exact discrete
velocity field ``u(t^n)``, with ``u^0 = u(0)`` starting from the exact
initial conditions. We say that the time integration scheme (definition of
``u^n``) is accurate to the order ``r`` if ``u^n = u(t^n) +
\mathcal{O}(\Delta t^r)`` for all ``n``.

IncompressibleNavierStokes provides a collection of explicit and implicit
Runge-Kutta methods, in addition to Adams-Bashforth Crank-Nicolson and one-leg
beta method time steppers.

The code is currently not adapted to time steppers from
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/),
but they may be integrated in the future.

```@docs
get_cache
AbstractODEMethod
runge_kutta_method
create_stepper
timestep
timestep!
```


## One-leg beta method

```@docs
OneLegMethod
```

## Runge-Kutta methods

```@docs
AbstractRungeKuttaMethod
ExplicitRungeKuttaMethod
RKMethods
LMWray3
```

### Explicit Methods

```@docs
RKMethods.FE11
RKMethods.SSP22
RKMethods.SSP42
RKMethods.SSP33
RKMethods.SSP43
RKMethods.SSP104
RKMethods.rSSPs2
RKMethods.rSSPs3
RKMethods.Wray3
RKMethods.RK56
RKMethods.DOPRI6
```

### Implicit Methods

```@docs
RKMethods.BE11
RKMethods.SDIRK34
RKMethods.ISSPm2
RKMethods.ISSPs3
```

### Half explicit methods

```@docs
RKMethods.HEM3
RKMethods.HEM3BS
RKMethods.HEM5
```

### Classical Methods

```@docs
RKMethods.GL1
RKMethods.GL2
RKMethods.GL3
RKMethods.RIA1
RKMethods.RIA2
RKMethods.RIA3
RKMethods.RIIA1
RKMethods.RIIA2
RKMethods.RIIA3
RKMethods.LIIIA2
RKMethods.LIIIA3
```

### Chebyshev methods

```@docs
RKMethods.CHDIRK3
RKMethods.CHCONS3
RKMethods.CHC3
RKMethods.CHC5
```

### Miscellaneous Methods

```@docs
RKMethods.Mid22
RKMethods.MTE22
RKMethods.CN22
RKMethods.Heun33
RKMethods.RK33C2
RKMethods.RK33P2
RKMethods.RK44
RKMethods.RK44C2
RKMethods.RK44C23
RKMethods.RK44P2
```

### DSRK Methods

```@docs
RKMethods.DSso2
RKMethods.DSRK2
RKMethods.DSRK3
```

### "Non-SSP" Methods of Wong & Spiteri

```@docs
RKMethods.NSSP21
RKMethods.NSSP32
RKMethods.NSSP33
RKMethods.NSSP53
```
