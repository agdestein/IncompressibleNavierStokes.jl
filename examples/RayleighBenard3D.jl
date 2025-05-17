# # Rayleigh-Bénard convection (3D)
#
# A hot and a cold plate generate a convection cell in a box.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Hardware
backend = IncompressibleNavierStokes.CPU()

## using CUDA, CUDSS
## backend = CUDABackend()

# Precision
T = Float32

# Output directory
outdir = joinpath(@__DIR__, "output", "RayleighBenard3D")

# Temperature equation
temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e7),
    Ge = T(1.0),
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
n = 60
x = (
    LinRange(T(0), T(π), 2n),
    tanh_grid(T(0), T(1), n, T(1.2)),
    tanh_grid(T(0), T(1), n, T(1.2)),
)
setup = Setup(;
    x,
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC()),
    ),
    Re = 1 / temperature.α1,
    temperature,
    backend,
);

plotgrid(x[1], x[2])
plotgrid(x[2], x[3])

# This will factorize the Laplace matrix
@time psolver = psolver_direct(setup)

# Initial conditions
ustart = velocityfield(setup, (dim, x, y, z) -> zero(x); psolver);
tempstart =
    temperaturefield(setup, (x, y, z) -> one(x) / 2 + sin(20 * x) * sinpi(20 * y) / 100);

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(10)),
    method = RKMethods.RK33C2(; T),
    ## Δt = T(1e-2),
    psolver,
    processors = (;
        ## anim = animator(;
        ##     path = "$outdir/RB3D.mp4",
        rtp = realtimeplotter(;
            setup,
            nupdate = 1,
            levels = LinRange(-T(5), T(1), 10),
            fieldname = :eig2field,
            ## levels = LinRange{T}(0.01, 0.99, 10),
            ## fieldname = :temperature,
        ),
        ## vtk = vtk_writer(;
        ##     setup,
        ##     dir = joinpath(outdir, "RB3D_$n"),
        ##     nupdate = 20,
        ##     ## fieldnames = (:velocity, :temperature, :eig2field)
        ##     fieldnames = (:temperature,),
        ## ),
        log = timelogger(; nupdate = 100),
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
