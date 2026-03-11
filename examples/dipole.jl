# # Two-dimensional dipole
#
# Study of a staggered fourth‐order compact scheme for unsteady incompressible viscous flows 
# Knikker, 2009, International Journal for Numerical Methods in Fluids 
# <https://onlinelibrary.wiley.com/doi/10.1002/fld.1854>

using IncompressibleNavierStokes
using GLMakie
using CUDA

getbackend() = CUDA.functional() ? CUDABackend() : IncompressibleNavierStokes.KernelAbstractions.CPU()

# Parameters
n = 1024                 # Number of grid points in each direction
lims = -1.0, 1.0         # Domain limits
viscosity = 1 / 2500     # Viscosity
tlims = 0.0, 1.0         # Simulation time
nplot = 100              # How often to plot
nlog = 100               # How often to log
dipole() = (;
    r0 = 0.1,            # Initial vortex radius
    x1 = 0.0, y1 = 0.1,  # Position of first vortex
    x2 = 0.0, y2 = -0.1, # Position of second vortex
    initialenergy = 2.0, # Initial kinetic energy
)

# Setup
ax = range(lims..., n + 1);
setup = Setup(;
    x = (ax, ax),
    boundary_conditions = (;
        u = (
            (DirichletBC(), DirichletBC()),
            (DirichletBC(), DirichletBC()),
            # (PeriodicBC(), PeriodicBC()),
            # (PeriodicBC(), PeriodicBC()),
        ),
    ),
    backend = getbackend(),
)

# psolver = default_psolver(setup)
psolver = psolver_transform(setup)

function U(dim, x, y)
    @inline
    (; x1, x2, y1, y2, r0) = dipole()
    r1 = (x - x1)^2 + (y - y1)^2
    r2 = (x - x2)^2 + (y - y2)^2
    d1 = ifelse(dim == 1, y - y1, x - x1)
    d2 = ifelse(dim == 1, y - y2, x - x2)
    return -d1 * exp(-r1 / r0^2) + d2 * exp(-r2 / r0^2)
end

u = velocityfield(setup, U; psolver)

# Scale velocity to have the correct initial kinetic energy
let
    (; initialenergy) = dipole()
    L = lims[2] - lims[1]
    u1 = u[setup.Iu[1], 1]
    u2 = u[setup.Iu[2], 2]
    kin = sum(abs2, u1) * L^2 / length(u1) + sum(abs2, u2) * L^2 / length(u2)
    @. u *= sqrt(initialenergy) / sqrt(kin)
end

# Solve unsteady problem
state, outputs = solve_unsteady(;
    setup,
    start = (; u),
    tlims,
    params = (; viscosity),
    psolver,
    processors = (
        rtp = realtimeplotter(;
            setup,
            plot = fieldplot,
            docolorbar = false,
            nupdate = nplot,
        ),
        log = timelogger(; nupdate = nlog),
    ),
);
