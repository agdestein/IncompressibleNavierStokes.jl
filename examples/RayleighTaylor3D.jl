# # Rayleigh-Taylor instability in 2D
#
# Two fluids with different temperatures start mixing.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Hardware
backend = IncompressibleNavierStokes.CPU()

## using CUDA, CUDSS
## backend = CUDABackend()

# Precision
T = Float64

# Output directory
outdir = joinpath(@__DIR__, "output", "RayleighTaylor3D")

# Temperature equation
temperature = temperature_equation(;
    Pr = T(0.71),
    Ra = T(1e6),
    Ge = T(1.0),
    dodissipation = true,
    boundary_conditions = (
        (SymmetricBC(), SymmetricBC()),
        (SymmetricBC(), SymmetricBC()),
        (SymmetricBC(), SymmetricBC()),
    ),
    gdir = 3,
    nondim_type = 1,
)

# Setup
n = 80
x = LinRange(T(0), T(1), n + 1), LinRange(T(0), T(1), n + 1), LinRange(T(0), T(2), 2n + 1)
setup = Setup(;
    x,
    boundary_conditions = (
        (DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC()),
    ),
    Re = 1 / temperature.α1,
    temperature,
    backend,
);

# This will factorize the Laplace matrix
@time psolver = psolver_direct(setup)

# Initial conditions
ustart = velocityfield(setup, (dim, x, y, z) -> zero(x); psolver);
tempstart = temperaturefield(setup, (x, y, z) -> (1 + sinpi(x / 20) * sinpi(y) > z));

fieldplot(
    (; u = ustart, temp = tempstart, t = T(0));
    ## state;
    setup,
    levels = LinRange{T}(0.8, 1, 5),
    ## levels = LinRange(-T(5), T(1), 10),
    ## fieldname = :eig2field,
    fieldname = :temperature,
    size = (400, 600),
)

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(40)),
    Δt = T(1e-2),
    psolver,
    processors = (;
        ## anim = animator(;
        ##     path = "$outdir/RT3D.mp4",
        rtp = realtimeplotter(;
            setup,
            nupdate = 20,
            fieldname = :eig2field,
            levels = LinRange(-T(5), T(1), 10),
            ## fieldname = :temperature,
            ## levels = LinRange{T}(0, 1, 10),
            size = (400, 600),
        ),
        ## vtk = vtk_writer(;
        ##     setup,
        ##     nupdate = 10,
        ##     dir = outdir,
        ##     fieldnames = (:velocity, :pressure, :temperature),
        ##     psolver,
        ## ),
        log = timelogger(; nupdate = 400),
    ),
);

# Check distribution of vortex structures
# for choosing plot levels
field = IncompressibleNavierStokes.eig2field(state.u, setup)[setup.grid.Ip]
hist(vec(Array(log.(max.(eps(T), .-field)))))

# Plot temperature field
fieldplot(state; setup, fieldname = :temperature)
