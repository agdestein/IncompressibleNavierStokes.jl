# # Rayleigh-Taylor instability in 2D
#
# Two fluids with different temperatures start mixing.

#md using CairoMakie
using WGLMakie #!md
using IncompressibleNavierStokes

# Output directory for saving results
outdir = joinpath(@__DIR__, "output", "RayleighTaylor2D")

# Precision
T = Float64

# Grid
n = 50
x = tanh_grid(T(0), T(1), n, T(1.5)), tanh_grid(T(0), T(2), 2n, T(1.5))
plotgrid(x...; figure = (; size = (300, 600)))

# Setup
setup = Setup(;
    x,
    boundary_conditions = (;
        u = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
        temp = ((SymmetricBC(), SymmetricBC()), (SymmetricBC(), SymmetricBC())),
    ),
);

# Initial conditions
start = (;
    u = velocityfield(setup, (dim, x, y) -> zero(x)),
    temp = temperaturefield(setup, (x, y) -> one(x) * (1 + sinpi(x) / 50 > y)),
)

# Solve equation
state, outputs = solve_unsteady(;
    force! = boussinesq!, # Solve the Boussinesq equations
    setup,
    start,
    tlims = (T(0), T(10)),
    params = (;
        viscosity = T(1e-3),
        gravity = T(1.0),
        gdir = 2, # Gravity acts in the y-direction
        conductivity = T(1e-3),
        dodissipation = true,
    ),
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
