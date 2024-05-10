#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Hardware
ArrayType = Array

## using CUDA, CUDSS
## ArrayType = CuArray

# Precision
T = Float64

# Grid
n = 50
x = tanh_grid(T(0), T(1), n, T(1.5))
y = tanh_grid(T(0), T(2), 2n, T(1.5))
plotgrid(x, y)

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
setup = Setup(
    x,
    y;
    boundary_conditions = ((DirichletBC(), DirichletBC()), (DirichletBC(), DirichletBC())),
    Re = 1 / temperature.α1,
    temperature,
);

# Initial conditions
ustart = create_initial_conditions(setup, (dim, x, y) -> zero(x));
(; xp) = setup.grid;
## T0(x, y) = one(x) * (1 > y);
T0(x, y) = one(x) * (1 + sinpi(x) > y); ## Perturbation
tempstart = T0.(xp[1], xp[2]');

# Solve equation
state, outputs = solve_unsteady(;
    setup,
    ustart,
    tempstart,
    tlims = (T(0), T(20)),
    Δt = T(5e-3),
    processors = (;
        rtp = realtimeplotter(;
            setup,
            nupdate = 10,
            fieldname = :temperature,
            size = (400, 600),
        ),
        log = timelogger(; nupdate = 100),
    ),
);

# Results
outputs.rtp
