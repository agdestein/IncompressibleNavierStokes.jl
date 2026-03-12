# # Rayleigh-Bénard convection (3D)
#
# A hot and a cold plate generate a convection cell in a box.

#md using CairoMakie
using WGLMakie #!md
using IncompressibleNavierStokes
## using CUDA, CUDSS

# Precision
T = Float64

# Output directory
outdir = joinpath(@__DIR__, "output", "RayleighBenard3D")

# Setup
n = 64
# x = (
#     range(T(0), T(π), 2n),
#     tanh_grid(T(0), T(1), n, T(1.2)),
#     tanh_grid(T(0), T(1), n, T(1.2)),
# )
x = (range(T(0), T(π), 2n + 1), range(T(0), T(1), n + 1), range(T(0), T(1), n + 1))
setup = Setup(;
    x,
    boundary_conditions = (;
        u = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), DirichletBC()),
            (DirichletBC(), DirichletBC()),
        ),
        temp = (
            (PeriodicBC(), PeriodicBC()),
            (SymmetricBC(), SymmetricBC()),
            (DirichletBC(T(1)), DirichletBC(T(0))),
        ),
    ),
    ## backend = CUDABackend()
);

plotgrid(x[1], x[2])

#-

plotgrid(x[2], x[3])

# AMGX solver (for NVidia GPUs)
## AMGX_stuff = amgx_setup();
## psolver = psolver_cg_AMGX(setup; stuff = AMGX_stuff);

# Direct pressure solver
## @time psolver = default_psolver(setup);

# Discrete transform solver (FFT/DCT)
psolver = psolver_transform(setup);

# Initial conditions
start = (;
    u = velocityfield(setup, (dim, x, y, z) -> zero(x); psolver),
    temp = temperaturefield(
        setup,
        (x, y, z) -> one(x) / 2 + sin(20 * x) * sinpi(20 * y) / 100,
    ),
);

# Solve equation
state, outputs = solve_unsteady(;
    force! = boussinesq!, # Solve the Boussinesq equations
    setup,
    start,
    tlims = (T(0), T(1)),
    psolver,
    params = (;
        viscosity = T(2.5e-4),
        gravity = T(1.0),
        gdir = 3, # Gravity in z-direction
        conductivity = T(2.5e-4),
        dodissipation = true,
    ),
    processors = (;
        ## rtp = realtimeplotter(;
        ##     setup,
        ##     nupdate = 1,
        ##     levels = LinRange(0.01 |> T, 0.99 |> T, 10),
        ##     fieldname = :temperature,
        ## ),
        ## vtk = vtk_writer(;
        ##     setup,
        ##     dir = joinpath(outdir, "RB3D_$n"),
        ##     nupdate = 20,
        ##     ## fieldnames = (:velocity, :temperature, :eig2field)
        ##     fieldnames = (:temperature,),
        ## ),
        log = timelogger(; nupdate = 1),
    ),
);
