![Logo](docs/src/assets/logo_text_dots.png)

# IncompressibleNavierStokes

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://agdestein.github.io/IncompressibleNavierStokes.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://agdestein.github.io/IncompressibleNavierStokes.jl/dev)
[![Build Status](https://github.com/agdestein/IncompressibleNavierStokes.jl/workflows/CI/badge.svg)](https://github.com/agdestein/IncompressibleNavierStokes.jl/actions)
[![Coverage](https://codecov.io/gh/agdestein/IncompressibleNavierStokes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/agdestein/IncompressibleNavierStokes.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package implements energy-conserving solvers for the incompressible Navier-Stokes
equations on a staggered cartesian grid. It is based on the Matlab package
[INS2D](https://github.com/bsanderse/INS2D)/[INS3D](https://github.com/bsanderse/INS3D).


## Installation

To install IncompressibleNavierStokes, clone the repository:

```sh
git clone https://github.com/agdestein/IncompressibleNavierStokes.jl
```

Then run the following command to install all dependencies from the [General
registry](https://github.com/JuliaRegistries/General) (as specified in
`Project.toml`):

```sh
cd IncompressibleNavierStokes
julia --project -e 'using Pkg; Pkg.instantiate()'
```

See the
[Documentation](https://agdestein.github.io/IncompressibleNavierStokes.jl/dev/generated/LidDrivenCavity2D/)
for examples of some typical workflows. More examples can be found in the
`examples` directory.

## Demo

The following example code simulates an actuator using a negative body force on
a small rectangle with an unsteady inflow.

```julia
using IncompressibleNavierStokes
using GLMakie

# Floating point type for simulations
T = Float64

# Models
viscosity_model = LaminarModel{T}(; Re = 100)

# Boundary conditions
f = 0.5
u_bc(x, y, t) = x ≈ 0.0 ? cos(π / 6 * sin(f * t)) : 0.0
v_bc(x, y, t) = x ≈ 0.0 ? sin(π / 6 * sin(f * t)) : 0.0
dudt_bc(x, y, t) = x ≈ 0.0 ? -π / 6 * f * cos(f * t) * sin(π / 6 * sin(f * t)) : 0.0
dvdt_bc(x, y, t) = x ≈ 0.0 ? π / 6 * f * cos(f * t) * cos(π / 6 * sin(f * t)) : 0.0
bc = create_boundary_conditions(
    u_bc,
    v_bc;
    dudt_bc,
    dvdt_bc,
    bc_unsteady = true,
    bc_type = (;
        u = (; x = (:dirichlet, :pressure), y = (:symmetric, :symmetric)),
        v = (; x = (:dirichlet, :symmetric), y = (:pressure, :pressure)),
    ),
    T,
)

# Grid
x = stretched_grid(0.0, 10.0, 200)
y = stretched_grid(-2.0, 2.0, 80)
grid = create_grid(x, y; bc, T);

# Body force
xc, yc = 2.0, 0.0 # Disk center
D = 1.0 # Disk diameter
δ = 0.11 # Disk thickness
Cₜ = 0.01 # Thrust coefficient
inside(x, y) = abs(x - xc) ≤ δ / 2 && abs(y - yc) ≤ D / 2
bodyforce_u(x, y) = -Cₜ * inside(x, y)
bodyforce_v(x, y) = 0.0
force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)

# Build setup and assemble operators
setup = Setup(; viscosity_model, grid, force, bc)

# Time interval
t_start, t_end = tlims = (0.0, 16π)

# Initial conditions (extend inflow)
initial_velocity_u(x, y) = 1.0
initial_velocity_v(x, y) = 0.0
initial_pressure(x, y) = 0.0
V₀, p₀ = create_initial_conditions(
    setup,
    t_start;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure,
);

# Solve unsteady problem
problem = UnsteadyProblem(setup, V₀, p₀, tlims)
V, p = solve_animate(
    problem, RK44P2();
    Δt = 4π / 200,
    filename = "vorticity.gif",
)
```

The resulting animation is shown below.

![Vorticity](assets/vorticity.gif)
