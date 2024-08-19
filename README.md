![Logo](docs/src/public/logo_text_dots.png#gh-light-mode-only)
![Logo](docs/src/public/logo_text_dots_dark_mode.png#gh-dark-mode-only)

# IncompressibleNavierStokes

| Documentation | Workflows | Code coverage | Quality assurance |
| :-----------: | :-------: | :-----------: | :---------------: |
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://agdestein.github.io/IncompressibleNavierStokes.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://agdestein.github.io/IncompressibleNavierStokes.jl/dev) | [![Build Status](https://github.com/agdestein/IncompressibleNavierStokes.jl/workflows/CI/badge.svg)](https://github.com/agdestein/IncompressibleNavierStokes.jl/actions) | [![Coverage](https://codecov.io/gh/agdestein/IncompressibleNavierStokes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/agdestein/IncompressibleNavierStokes.jl) | [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) |

This package implements energy-conserving solvers for the incompressible Navier-Stokes
equations on a staggered Cartesian grid. It is based on the Matlab package
[INS2D](https://github.com/bsanderse/INS2D)/[INS3D](https://github.com/bsanderse/INS3D). The simulations can be run on the single/multithreaded CPUs or Nvidia GPUs.

This package also provides experimental support for neural closure models for
large eddy simulation.

## Installation

To install IncompressibleNavierStokes, open up a Julia-REPL, type `]` to get
into Pkg-mode, and type:

```julia-repl
(v1.10) pkg> add IncompressibleNavierStokes
```

which will install the package and all dependencies to your local environment.
Note that IncompressibleNavierStokes requires Julia version `1.9` or above.

See the
[Documentation](https://agdestein.github.io/IncompressibleNavierStokes.jl/dev/generated/LidDrivenCavity2D/)
for examples of some typical workflows. More examples can be found in the
[`examples`](examples) directory.

## Source code for paper

See [here](./lib/PaperDC) for the source code used in the paper
[Discretize first, filter next: learning divergence-consistent closure models for large-eddy simulation](https://arxiv.org/abs/2403.18088).

## Gallery

The velocity and pressure fields may be visualized in a live session using
[Makie](https://github.com/JuliaPlots/Makie.jl). Alternatively,
[ParaView](https://www.paraview.org/) may be used, after exporting individual
snapshot files using the `save_vtk` function, or the full time series using the
`VTKWriter` processor.

<table>
    <tbody>
        <tr>
            <td style="width: 25%;"><a href="examples/Actuator2D.jl"><video autoplay loop src="docs/src/public/Actuator2D.mp4"></video></a></td>
            <td style="width: 25%;"><a href="examples/BackwardFacingStep2D.jl"><img src="docs/src/public/BackwardFacingStep2D.png"></a></td>
            <td style="width: 25%;"><a href="examples/DecayingTurbulence2D.jl"><video autoplay loop src="docs/src/public/DecayingTurbulence2D.mp4"></video></a></td>
            <td style="width: 25%;"><a href="examples/TaylorGreenVortex2D.jl"><img src="docs/src/public/TaylorGreenVortex2D.png"></a></td>
        </tr>
        <tr>
            <td align="center">Actuator (2D)</td>
            <td align="center">Backward facing step (2D)</td>
            <td align="center">Decaying turbulence (2D)</td>
            <td align="center">Taylor-Green vortex (2D)</td>
        </tr>
        <tr>
            <td style="width: 25%;"><a href="examples/Actuator3D.jl"><img src="docs/src/public/Actuator3D.png"></a></td>
            <td style="width: 25%;"><a href="examples/BackwardFacingStep3D.jl"><img src="docs/src/public/BackwardFacingStep3D.png"></a></td>
            <td style="width: 25%;"><a href="examples/DecayingTurbulence3D.jl"><img src="docs/src/public/DecayingTurbulence3D.png"></a></td>
            <td style="width: 25%;"><a href="examples/TaylorGreenVortex3D.jl"><img src="docs/src/public/TaylorGreenVortex3D.png"></a></td>
        </tr>
        <tr>
            <td align="center">Actuator (3D)</td>
            <td align="center">Backward facing step (3D)</td>
            <td align="center">Decaying turbulence (3D)</td>
            <td align="center">Taylor-Green vortex (3D)</td>
        </tr>
        <tr>
            <td style="width: 25%;"><a href="examples/RayleighBenard2D.jl"><video autoplay loop src="docs/src/public/RayleighBenard2D.mp4"></video></a></td>
            <td style="width: 25%;">
              <!-- <a href="examples/RayleighBenard3D.jl"><img src="docs/src/public/RayleighBenard3D.png"></a> -->
            </td>
            <td style="width: 25%;"><a href="examples/RayleighTaylor2D.jl"><video autoplay loop src="docs/src/public/RayleighTaylor2D.mp4"></video></a></td>
            <td style="width: 25%;">
              <!-- <a href="examples/RayleighTaylor3D.jl"><img src="docs/src/public/RayleighTaylor3D.png"></a> -->
            </td>
        </tr>
        <tr>
            <td align="center">Rayleigh-Bénard (2D)</td>
            <td align="center">Rayleigh-Bénard (3D)</td>
            <td align="center">Rayleigh-Taylor (2D)</td>
            <td align="center">Rayleigh-Taylor (3D)</td>
        </tr>
    </tbody>
</table>


IncompressibleNavierStokes also supports adding a temperature equation.

https://github.com/agdestein/IncompressibleNavierStokes.jl/assets/40632532/a264054b-641f-4bd4-b25c-693b9794e41d

## Demo

The following example code using a negative body force on a small rectangle
with an unsteady inflow. It simulates a wind turbine (actuator) under varying
wind conditions.

Make sure to have the `GLMakie` and `IncompressibleNavierStokes` installed:

```julia
using Pkg
Pkg.add(["GLMakie", "IncompressibleNavierStokes"])
```

Then run run the following code to make a short animation:

```julia
using GLMakie
using IncompressibleNavierStokes

# Setup
setup = Setup(
    tanh_grid(0.0, 2.0, 200, 1.2),
    tanh_grid(0.0, 1.0, 100, 1.2);
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    temperature = temperature_equation(;
        Pr = 0.71,
        Ra = 1e7,
        Ge = 1.0,
        boundary_conditions = (
            (SymmetricBC(), SymmetricBC()),
            (DirichletBC(1.0), DirichletBC(0.0)),
        ),
    ),
)

# Initial conditions
U0(dim, x, y) = zero(x)
T0(x, y) = 1 / 2 + sinpi(30 * x) / 100
ustart = create_initial_conditions(setup, U0)
tempstart = IncompressibleNavierStokes.apply_bc_temp(T0.(setup.grid.xp[1], setup.grid.xp[2]'), 0.0, setup)

# Solve equation
solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (0.0, 30.0),
    Δt = 0.02,
    processors = (;
        anim = animator(;
            setup,
            path = "temperature.mp4",
            fieldname = :temperature,
            colorrange = (0.0, 1.0),
            size = (900, 500),
            colormap = :seaborn_icefire_gradient,
            nupdate = 5,
        ),
    ),
)
```

The resulting animation is shown below.

https://github.com/agdestein/IncompressibleNavierStokes.jl/assets/40632532/6ee09a03-1674-46e0-843c-000f0b9b9527

## Similar projects

- [WaterLily.jl](https://github.com/weymouth/WaterLily.jl/)
  Incompressible solver with immersed boundaries
- [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl):
  Ocean simulations
- [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl):
  Atmospheric simulations
- [Trixi.jl](https://github.com/trixi-framework/Trixi.jl):
  High order solvers for various hyperbolic equations
- [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl):
  Finite element discretizations
- [Gridap.jl](https://github.com/gridap/Gridap.jl):
  Finite element discretizations
- [FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl):
  Pseudo-spectral discretizations
