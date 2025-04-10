# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

using WGLMakie
# using GLMakie
# using CairoMakie
using IncompressibleNavierStokes
using FFTW
using Adapt

# Output directory
output = "output/Channel"

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

using CUDA;
T = Float32;
ArrayType = CuArray;
CUDA.allowscalar(false);
device = x -> adapt(CuArray{T}, x)

set_theme!(; GLMakie = (; scalefactor = 1.5))

# Reynolds number
Re = T(4_000)

# Boundary conditions
boundary_conditions = (
    ## x left, x right
    (PeriodicBC(), PeriodicBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC()),
)

getdir(x = rand(T)) = T[cospi(2x), sinpi(2x)]
createbodyforce((Ax, Ay), (a, b), r) = function bodyforce(dim, x, y, t)
    A = Ax * (dim() == 1) + Ay * (dim() == 2)
    R = (x - a)^2 / r^2 + (y - b)^2 / r^2
    A * max(0, 1 - R)
end
createmanyforce(forces...) = function (dim, x, y, t)
    out = zero(x)
    for f in forces
        out += f(dim, x, y, t)
    end
    out
end

bodyforce = createmanyforce(
    createbodyforce(2 * getdir(+0.00), T[1.0, +0.0], T(0.6)),
    createbodyforce(4 * getdir(-0.10), T[2.0, +0.5], T(0.2)),
    createbodyforce(3 * getdir(+0.05), T[3.5, -0.3], T(0.3)),
    createbodyforce(4 * getdir(-0.05), T[4.5, +0.1], T(0.2)),
    createbodyforce(4 * getdir(-0.45), T[6.5, +0.5], T(0.2)),
    createbodyforce(3 * getdir(+0.05), T[7.5, +0.0], T(0.3)),
    createbodyforce(4 * getdir(-0.50), T[9.0, -0.5], T(0.2)),
)

bodyforce = createmanyforce(
    createbodyforce(4 * getdir(+0.10), T[1.0, +0.4], T(0.3)),
    createbodyforce(3 * getdir(-0.45), T[2.3, -0.1], T(0.2)),
    createbodyforce(4 * getdir(+0.05), T[3.0, +0.2], T(0.2)),
    createbodyforce(2 * getdir(-0.02), T[5.0, -0.1], T(0.5)),
    createbodyforce(4 * getdir(-0.45), T[7.0, +0.4], T(0.2)),
    createbodyforce(3 * getdir(+0.05), T[7.5, -0.5], T(0.3)),
    createbodyforce(4 * getdir(+0.00), T[9.0, +0.0], T(0.2)),
)

# A 2D grid is a Cartesian product of two vectors. Here we refine the grid near
# the walls.
n = 64
# n = 128
x = LinRange(T(0), T(10), 8n + 1)
y = LinRange(-T(1), T(1), 2n + 1)
plotgrid(x, y)

bodyforce.([IncompressibleNavierStokes.Dimension(1)], x, y', T(0)) |> heatmap
bodyforce.([IncompressibleNavierStokes.Dimension(2)], x, y', T(0)) |> heatmap

# Build setup and assemble operators
setup = Setup(x, y; Re, bodyforce, boundary_conditions, ArrayType);

psolver = DirectPressureSolver(setup);

# Initial conditions
u₀ = create_initial_conditions(
    setup,
    (dim, x, y) -> (dim() == 1) * 3 * (1 + y) * (1 - y);
    # (dim, x, y) -> zero(x);
    psolver,
);
u = u₀;

# Solve unsteady problem
state, outputs = solve_unsteady(
    setup,
    # u₀,
    u,
    (T(0), T(4.0));
    Δt = T(0.005),
    psolver,
    processors = (
        rtp = realtimeplotter(;
            setup,
            # type = contourf,
            # plot = fieldplot,
            # fieldname = :velocity,
            # plot = energy_history_plot,
            ## plot = energy_spectrum_plot,
            nupdate = 1,
            size = (1200, 600),
            docolorbar = false,
        ),
        ## anim = animator(; setup, path = "$output/vorticity.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = output, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 1),
    ),
);
(; u) = state;

using CairoMakie
GLMakie.activate!()
CairoMakie.activate!()

# Plot pressure
fieldplot(
    state;
    setup,
    psolver,
    # type = contourf,
    # fieldname = :pressure,
    fieldname = :velocity,
    # fieldname = :vorticity,
    docolorbar = false,
    # size = (1200, 350),
    # size = (900, 250),
    size = (650, 200),
)

name = "vorticity_10.pdf"
name = "velocity_10.pdf"
save(name, current_figure())
run(`mv $name ../SupervisedClosure/figures/channel/`)

(; Ip, Iu, xp) = setup.grid

fig = with_theme(; palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])) do
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x")
    lines!(ax, Array(xp[1][2:(end-1)]), u[2][Iu[2]][:, 32] |> Array)
    fig
end
name = "midline_10.pdf"

IncompressibleNavierStokes.kinetic_energy(u, setup)[Ip] |> Array |> heatmap
sum(IncompressibleNavierStokes.kinetic_energy(u, setup)[Ip]; dims = 2) |> lines
sum(IncompressibleNavierStokes.kinetic_energy(u, setup)[Ip]; dims = 1)[:] |> Array |> lines

e = IncompressibleNavierStokes.kinetic_energy(u, setup)[Ip]
e[:, 40]

ehat = fft(e[:, n])[1:5n] |> x -> abs.(x) |> Array
ehat2 = fft(e[:, 60])[1:5n] |> x -> abs.(x) |> Array
lines(ehat; axis = (; xscale = log10, yscale = log10))
lines!(ehat2)

fig = Figure()
Axis(fig[1, 1])
u[2][Iu[2]][:, 32] |> Array |> lines!
# u[2][Iu[2]][:, 40] |> Array |> lines!

fig = Figure()
Axis(fig[1, 1])
u[1][Iu[1]][20, :] |> Array |> lines!
u[1][Iu[1]][40, :] |> Array |> lines!
u[1][Iu[1]][300, :] |> Array |> lines!

# ## Post-process
#
# We may visualize or export the computed fields

# Export to VTK
save_vtk(setup, state.u, state.p, "$output/solution")

Makie.available_gradients()

#######################################################################

bodyforce_train = createmanyforce(
    createbodyforce(2 * getdir(+0.00), T[1.0, +0.0], T(0.6)),
    createbodyforce(4 * getdir(-0.10), T[2.0, +0.5], T(0.2)),
    createbodyforce(3 * getdir(+0.05), T[3.5, -0.3], T(0.3)),
    createbodyforce(4 * getdir(-0.05), T[4.5, +0.1], T(0.2)),
    createbodyforce(4 * getdir(-0.45), T[6.5, +0.5], T(0.2)),
    createbodyforce(3 * getdir(+0.05), T[7.5, +0.0], T(0.3)),
    createbodyforce(4 * getdir(-0.50), T[9.0, -0.5], T(0.2)),
)

bodyforce_test = createmanyforce(
    createbodyforce(4 * getdir(+0.10), T[1.0, +0.4], T(0.3)),
    createbodyforce(3 * getdir(-0.45), T[2.3, -0.1], T(0.2)),
    createbodyforce(4 * getdir(+0.05), T[3.0, +0.2], T(0.2)),
    createbodyforce(2 * getdir(-0.02), T[5.0, -0.1], T(0.5)),
    createbodyforce(4 * getdir(-0.45), T[7.0, +0.4], T(0.2)),
    createbodyforce(3 * getdir(+0.05), T[7.5, -0.5], T(0.3)),
    createbodyforce(4 * getdir(+0.00), T[9.0, +0.0], T(0.2)),
)

# Parameters
# nles = 50
# nles = 64
# nles = 128
# ndns = 200
nles = 32
ndns = 128
params = (;
    D = 2,
    Re = T(4_000),
    lims = ((T(0), T(10)), (T(-1), T(1))),
    nles = [(8nles, 2nles)],
    ndns = (8ndns, 2ndns),
    # tburn = T(0.1),
    tsim = T(10),
    Δt = T(1e-3),
    savefreq = 5,
    ArrayType,
    boundary_conditions = ((PeriodicBC(), PeriodicBC()), (DirichletBC(), DirichletBC())),
    PSolver = DirectPressureSolver,
    icfunc = (setup, psolver) -> create_initial_conditions(
        setup,
        (dim, x, y) -> (dim() == 1) * 3 * (1 + y) * (1 - y);
        # (dim, x, y) -> zero(x);
        psolver,
    ),
)

# Create LES data from DNS
data_train = [create_les_data(T; params..., bodyforce = bodyforce_train) for _ = 1:1];
data_test = [create_les_data(T; params..., bodyforce = bodyforce_test) for _ = 1:1];

data_train[1].u[1][1][1]

# Build LES setup and assemble operators
x = ntuple(α -> LinRange(params.lims[α]..., params.nles[1][α] + 1), params.D)
setup = Setup(x...; params.boundary_conditions, params.Re, ArrayType);

# Uniform periodic grid
psolver = params.PSolver(setup);

# Inspect data
field = data_train[1].u[1];

u = device(field[1])
# u = device(field[201])
# u = device(field[1201])
# u = device(field[2001])
o = Observable((; u, t = nothing))
fieldplot(
    o;
    setup,
    # fieldname = :velocity,
    # fieldname = 2,
)

# energy_spectrum_plot(o; setup)
for i = 1:length(field)
    o[] = (; o[]..., u = device(field[i]))
    sleep(0.001)
end

io_train = create_io_arrays(data_train, [setup]);
io_test = create_io_arrays(data_test, [setup]);

closure, θ₀ = cnn(;
    setup,
    radii = [2, 2, 2, 2],
    channels = [5, 5, 5, params.D],
    activations = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
    rng,
);
closure.chain

sample = io_train[1].u[:, :, :, 1:5]
closure(sample, θ₀) |> size
