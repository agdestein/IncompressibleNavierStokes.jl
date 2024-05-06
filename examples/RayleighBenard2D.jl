#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

using CUDA, CUDSS
ArrayType = CuArray

outdir = joinpath(@__DIR__, "output")

temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e7),
    Ge = T(0.1),
    dodissipation = true,
    boundary_conditions = (
        (SymmetricBC(), SymmetricBC()),
        (DirichletBC(1.0), DirichletBC(0.0)),
    ),
    gdir = 2,
    nondim_type = 1,
)

n = 64
x = LinRange(T(0), T(1), n + 1)
y = LinRange(T(0), T(1), n + 1)
setup = Setup(
    x,
    y;
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    Re = 1 / temperature.α1,
    temperature,
    ArrayType,
);
ustart = create_initial_conditions(setup, (dim, x, y) -> zero(x));
(; xp) = setup.grid;
tempstart = @. max(-sin($(T(3π)) * xp[1]) / 100, 0) + (1 - xp[2]');

state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (0.0, 100.0),
    Δt = 1e-2,
    processors = (;
        # rtp = realtimeplotter(;
        anim = animator(;
            path = "$outdir/RB2D.mp4",
            setup,
            nupdate = 100,
            fieldname = :temperature,
            colorrange = (0.0, 1.0),
            # displayupdates = true,
            size = (600, 500),
        ),
        log = timelogger(; nupdate = 20),
    ),
);
