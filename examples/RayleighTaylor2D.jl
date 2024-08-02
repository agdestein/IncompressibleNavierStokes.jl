# # Rayleigh-Taylor instability in 2D
#
# Two fluids with different temperatures start mixing.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory for saving results
outdir = joinpath(@__DIR__, "output", "RayleighTaylor2D")

# Hardware
ArrayType = Array

## using CUDA, CUDSS
## ArrayType = CuArray

# Precision
T = Float64

# Grid
n = 50
x = tanh_grid(T(0), T(1), n, T(1.5))
y = tanh_grid(T(0), T(2), 2n, T(1.5))
plotgrid(x, y; figure = (; size = (300, 600)))

# Setup
temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e6),
    Ge = T(1.0),
    dodissipation = true,
    boundary_conditions = ((SymmetricBC(), SymmetricBC()), (SymmetricBC(), SymmetricBC())),
    gdir = 2,
    nondim_type = 1,
)
setup = Setup(
    x,
    y;
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    Re = 1 / temperature.α1,
    temperature,
);

# Initial conditions
ustart = create_initial_conditions(setup, (dim, x, y) -> zero(x));
(; xp) = setup.grid;
## T0(x, y) = one(x) * (1 > y);
T0(x, y) = one(x) * (1 + sinpi(x) / 50 > y); ## Perturbation
tempstart = T0.(xp[1], xp[2]');

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(10)),
    Δt = T(5e-3),
    processors = (;
        rtp = realtimeplotter(;
            setup,
            nupdate = 20,
            fieldname = :temperature,
            size = (400, 600),
        ),
        log = timelogger(; nupdate = 200),
    ),
);

#md # ```@raw html
#md # <video src="/RayleighTaylor2D.mp4" controls="controls" autoplay="autoplay" loop="loop"></video>
#md # ```
