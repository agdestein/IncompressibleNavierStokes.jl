# Boundary conditions

Each boundary has exactly one type of boundary conditions. For periodic
boundary conditions, the opposite boundary must also be periodic.
The available boundary conditions are given below.

```@docs
PeriodicBC
DirichletBC
SymmetricBC
PressureBC
```

```@docs
IncompressibleNavierStokes.offset_p
IncompressibleNavierStokes.offset_u
```
