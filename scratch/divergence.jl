# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using LinearAlgebra

# Output directory
output = "output/DecayingTurbulence2D"

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Viscosity model
Re = T(10_000)

# A 2D grid is a Cartesian product of two vectors
nles = 256
comp = 2
ndns = nles * comp
lims = T(0), T(1)
D = 2
dns = Setup(ntuple(α -> LinRange(lims..., ndns + 1), D)...; Re, ArrayType);
les = Setup(ntuple(α -> LinRange(lims..., nles + 1), D)...; Re, ArrayType);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
psolver_dns = SpectralPressureSolver(dns);
psolver_les = SpectralPressureSolver(les);

# Create random initial conditions
u₀ = random_field(dns, T(0); psolver);

# Solve unsteady problem
state, outputs = solve_unsteady(
    dns,
    u₀,
    (T(0), T(0.1));
    Δt = T(1e-3),
    psolver_dns,
    processors = (
        rtp = realtimeplotter(; setup = dns, nupdate = 1),
        # ehist = realtimeplotter(;
        #     setup,
        #     plot = energy_history_plot,
        #     nupdate = 10,
        #     displayfig = false,
        # ),
        # espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        # anim = animator(; setup, path = "$output/solution.mkv", nupdate = 20),
        # vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        # field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);
(; u, t) = state;

ubar = IncompressibleNavierStokes.face_average(u, les, comp);
ubar = IncompressibleNavierStokes.volume_average(u, les, comp);
fieldplot((; u = ubar, t); setup = les)

IncompressibleNavierStokes.apply_bc_u!(ubar, t, les)
div = IncompressibleNavierStokes.divergence(ubar, les)[les.grid.Ip]

norm(div)
norm(ubar[1][les.grid.Iu[1]])
