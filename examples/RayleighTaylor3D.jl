#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Hardware
ArrayType = Array

## using CUDA, CUDSS
## ArrayType = CuArray

# Precision
T = Float64

# Output directory
outdir = joinpath(@__DIR__, "output")

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
x = LinRange(T(0), T(1), n + 1)
y = LinRange(T(0), T(1), n + 1)
z = LinRange(T(0), T(2), 2n + 1)
setup = Setup(
    x,
    y,
    z;
    boundary_conditions = (
        (DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC()),
        (DirichletBC(), DirichletBC()),
    ),
    Re = 1 / temperature.α1,
    temperature,
    ArrayType,
);

# This will factorize the Laplace matrix
@time psolver = psolver_direct(setup)

# Initial conditions
ustart = create_initial_conditions(setup, (dim, x, y, z) -> zero(x); psolver);
(; xp) = setup.grid;
xx = xp[1];
xy = reshape(xp[2], 1, :);
xz = reshape(xp[3], 1, 1, :);
tempstart = @. $(T(1)) * (1.0 + 0.05 * sin($(T(1.05 * π)) * xx) * sin($(T(π)) * xy) > xz);

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
        log = timelogger(; nupdate = 10),
    ),
);

field = IncompressibleNavierStokes.eig2field(state.u, setup)[setup.grid.Ip]
hist(vec(Array(log.(max.(eps(T), .-field)))))

fieldplot(state; setup, fieldname = :temperature)
