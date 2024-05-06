#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

using CUDA, CUDSS
T = Float32
ArrayType = CuArray

outdir = joinpath(@__DIR__, "output")

temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e6),
    # Ge = T(0.1),
    Ge = T(1.0),
    dodissipation = true,
    boundary_conditions = (
        (SymmetricBC(), SymmetricBC(), SymmetricBC()),
        (SymmetricBC(), SymmetricBC(), SymmetricBC()),
        (SymmetricBC(), SymmetricBC(), SymmetricBC()),
    ),
    gdir = 3,
    nondim_type = 1,
)

n = 80
x = LinRange(T(0), T(1), n + 1)
y = LinRange(T(0), T(1), n + 1)
z = LinRange(T(0), T(2), 2n + 1)
setup = Setup(
    x,
    y,
    z;
    boundary_conditions = (
        (DirichletBC(), DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC(), DirichletBC()),
    ),
    Re = 1 / temperature.α1,
    temperature,
    ArrayType,
);

psolver = psolver_direct(setup)

ustart = create_initial_conditions(setup, (dim, x, y, z) -> zero(x); psolver);
(; xp) = setup.grid;
xx = xp[1];
xy = reshape(xp[2], 1, :);
xz = reshape(xp[3], 1, 1, :);
tempstart = @. $(T(1)) * (1.0 + 0.05 * sin($(T(1.05 * π)) * xx) * sin($(T(π)) * xy) > xz);

fieldplot(
    (; u = ustart, temp = tempstart, t = T(0));
    # state;
    setup,
    levels = LinRange{T}(0.8, 1, 5),
    # levels = LinRange(-T(5), T(1), 10),
    # fieldname = :eig2field,
    fieldname = :temperature,
    size = (400, 600),
)

save("$outdir/RT3D_initial.png", current_figure())

state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (0.0, 40.0),
    # Δt = 5e-3,
    Δt = 1e-2,
    psolver,
    processors = (;
        # rtp = realtimeplotter(;
        anim = animator(;
            # path = "$outdir/RT3D_temp.mp4",
            path = "$outdir/RT3D_l2.mp4",
            setup,
            nupdate = 20,
            fieldname = :eig2field,
            # fieldname = :temperature,
            levels = LinRange(-T(5), T(1), 10),
            # levels = LinRange{T}(0, 1, 10),
            # displayupdates = true,
            size = (400, 600),
        ),
        log = timelogger(; nupdate = 10),
    ),
);

field = IncompressibleNavierStokes.eig2field(state.u, setup)[setup.grid.Ip]
# hist(vec(Array(log(max(eps(T), field)))
hist(vec(Array(log.(max.(eps(T), .-field)))))

fieldplot(state; setup, fieldname = :temperature)
