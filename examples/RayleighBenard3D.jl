#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Hardware
ArrayType = Array

## using CUDA, CUDSS
## ArrayType = CuArray

# Precision
T = Float32

# Output directory
outdir = joinpath(@__DIR__, "output")

# Temperature equation
temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e6),
    Ge = T(0.1),
    dodissipation = true,
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (SymmetricBC(), SymmetricBC()),
        (DirichletBC(T(1)), DirichletBC(T(0))),
    ),
    gdir = 3,
    nondim_type = 1,
)

# Setup
n = 40
x = LinRange(T(0), T(π), 2n)
y = tanh_grid(T(0), T(1), n, T(1.2))
z = tanh_grid(T(0), T(1), n, T(1.2))
setup = Setup(
    x,
    y,
    z;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC()),
    ),
    Re = 1 / temperature.α1,
    temperature,
    ArrayType,
);

plotgrid(x, y)
plotgrid(y, z)

# This will factorize the Laplace matrix
@time psolver = psolver_direct(setup)

# Initial conditions
ustart = create_initial_conditions(setup, (dim, x, y, z) -> zero(x); psolver);
(; xp) = setup.grid;
## T0(x, y, z) = 1 - z;
T0(x, y, z) = 1 - z + max(sin(8 * x) * sinpi(4 * y) / 100, 0) ; ## Perturbation
tempstart = T0.(xp[1], reshape(xp[2], 1, :), reshape(xp[3], 1, 1, :));

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(50)),
    method = RKMethods.RK33C2(; T),
    Δt = T(1e-2),
    psolver,
    processors = (;
        ## anim = animator(;
        ##     path = "$outdir/RB3D.mp4",
        ## rtp = realtimeplotter(;
        ##     setup,
        ##     nupdate = 50,
        ##     levels = LinRange(-T(5), T(1), 10),
        ##     fieldname = :eig2field,
        ##     ## levels = LinRange{T}(0, 1, 10),
        ##     ## fieldname = :temperature,
        ## ),
        vtk = vtk_writer(;
            setup,
            dir = joinpath(outdir, "RB3D_$n"),
            nupdate = 20,
            ## fieldnames = (:velocity, :temperature, :eig2field)
            fieldnames = (:temperature,)
        ),
        log = timelogger(; nupdate = 1),
    ),
);

field = IncompressibleNavierStokes.eig2field(state.u, setup)[setup.grid.Ip]
hist(vec(Array(log.(max.(eps(T), .-field)))))

fieldplot(
    ## (; u = ustart, temp = tempstart, t = T(0));
    state;
    setup,
    levels = LinRange{T}(0.5, 2, 5),
    ## levels = LinRange(-T(5), T(1), 10),
    ## fieldname = :eig2field,
    fieldname = :temperature,
)
