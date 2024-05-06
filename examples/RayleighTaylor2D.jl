#md using CairoMakie
using GLMakie #!md
using CairoMakie
using IncompressibleNavierStokes

using CUDA, CUDSS
T = Float32
ArrayType = CuArray

outdir = joinpath(@__DIR__, "output")

temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e6),
    Ge = T(1.0),
    # Ge = T(0.1),
    dodissipation = true,
    boundary_conditions = ((SymmetricBC(), SymmetricBC()), (SymmetricBC(), SymmetricBC())),
    gdir = 2,
    nondim_type = 1,
)

n = 128
x = LinRange(T(0), T(1), n + 1)
y = LinRange(T(0), T(2), 2n + 1)
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
tempstart = @. $(T(1)) * (1.0 + 0.05 * sin($(T(2π)) * xp[1]) > xp[2]');
tempstart = @. $(T(1)) * (1.0 + 0.05 * sin($(T(π)) * xp[1]) > xp[2]');
# @. tempstart += 0.3 * randn()

state, outputs = solve_unsteady(;
    setup,
    # ustart,
    # tempstart,
    ustart = state.u,
    tempstart = state.temp,
    tlims = (0.0, 40.0),
    Δt = 5e-3,
    processors = (;
        rtp = realtimeplotter(;
            # anim = animator(;
            path = "$outdir/RT2D.mp4",
            setup,
            nupdate = 200,
            fieldname = :temperature,
            displayupdates = true,
            size = (400, 600),
        ),
        log = timelogger(; nupdate = 50),
    ),
);

state.u .|> extrema
state.temp |> extrema
state.u[1]
state.u[2]
state.temp

fieldplot(state; setup, fieldname = :temperature)
save("toto.png", current_figure())
