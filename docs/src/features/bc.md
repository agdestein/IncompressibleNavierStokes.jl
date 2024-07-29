```@meta
CurrentModule = IncompressibleNavierStokes
```

# Boundary conditions

Each boundary has exactly one type of boundary conditions. For periodic
boundary conditions, the opposite boundary must also be periodic.
The available boundary conditions are given below.

```@docs
AbstractBC
PeriodicBC
DirichletBC
SymmetricBC
PressureBC
```

```@docs
offset_p
offset_u
```
