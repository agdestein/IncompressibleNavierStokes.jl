# Time discretization

The spatially discretized Navier-Stokes equations form a differential-algebraic
system, with an ODE for the velocity

```math
\Omega \frac{\mathrm{d} V_h}{\mathrm{d} t} = F(V_h) - (G p_h + y_G)
```

and an algebraic equation for the pressure

```math
A p_h = g.
```

## Runge-Kutta methods

See Sanderse [Sanderse2012](@cite) [Sanderse2013](@cite).
