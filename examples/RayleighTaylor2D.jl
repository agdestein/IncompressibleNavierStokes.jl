# # Rayleigh-Taylor instability in 2D
#
# Two fluids with different temperatures start mixing.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory for saving results
outdir = joinpath(@__DIR__, "output", "RayleighTaylor2D")

# Hardware
backend = IncompressibleNavierStokes.CPU()

## using CUDA, CUDSS
## backend = CUDABackend()

# Precision
T = Float64

# Grid
n = 50
x = tanh_grid(T(0), T(1), n, T(1.5)), tanh_grid(T(0), T(2), 2n, T(1.5))
plotgrid(x...; figure = (; size = (300, 600)))

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
setup = Setup(;
    x,
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    temperature,
);

# Initial conditions
start = (;
    u = velocityfield(setup, (dim, x, y) -> zero(x)),
    temp = temperaturefield(setup, (x, y) -> one(x) * (1 + sinpi(x) / 50 > y)),
)

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    start,
    tlims = (T(0), T(10)),
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
