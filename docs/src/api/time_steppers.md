```@meta
CurrentModule = IncompressibleNavierStokes
```

# API Reference -- time steppers

## ODE methods

```@docs
AdamsBashforthCrankNicolsonMethod
OneLegMethod
ExplicitRungeKuttaMethod
ImplicitRungeKuttaMethod
runge_kutta_method
```

## Explicit Methods

```@docs
FE11
SSP22
SSP42
SSP33
SSP43
SSP104
rSSPs2
rSSPs3
Wray3
RK56
DOPRI6
```

## Implicit Methods

```@docs
BE11
SDIRK34
ISSPm2
ISSPs3
```

## Half explicit methods

```@docs
HEM3
HEM3BS
HEM5
```

## Classical Methods

```@docs
GL1
GL2
GL3
RIA1
RIA2
RIA3
RIIA1
RIIA2
RIIA3
LIIIA2
LIIIA3
```

## Chebyshev methods

```@docs
CHDIRK3
CHCONS3
CHC3
CHC5
```

## Miscellaneous Methods

```@docs
Mid22
MTE22
CN22
Heun33
RK33C2
RK33P2
RK44
RK44C2
RK44C23
RK44P2
```

## DSRK Methods

```@docs
DSso2
DSRK2
DSRK3
```

## "Non-SSP" Methods of Wong & Spiteri

```@docs
NSSP21
NSSP32
NSSP33
NSSP53
```
