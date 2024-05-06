```@meta
CurrentModule = IncompressibleNavierStokes
```

# API Reference -- Runge-Kutta methods

```@docs
RKMethods
```

## Explicit Methods

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

## Implicit Methods

```@docs
RKMethods.BE11
RKMethods.SDIRK34
RKMethods.ISSPm2
RKMethods.ISSPs3
```

## Half explicit methods

```@docs
RKMethods.HEM3
RKMethods.HEM3BS
RKMethods.HEM5
```

## Classical Methods

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

## Chebyshev methods

```@docs
RKMethods.CHDIRK3
RKMethods.CHCONS3
RKMethods.CHC3
RKMethods.CHC5
```

## Miscellaneous Methods

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

## DSRK Methods

```@docs
RKMethods.DSso2
RKMethods.DSRK2
RKMethods.DSRK3
```

## "Non-SSP" Methods of Wong & Spiteri

```@docs
RKMethods.NSSP21
RKMethods.NSSP32
RKMethods.NSSP33
RKMethods.NSSP53
```
