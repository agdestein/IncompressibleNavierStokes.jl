# Little LSP hack to get function signatures, go    #src
# to definition etc.                                #src
if isdefined(@__MODULE__, :LanguageServer)          #src
    include("../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes               #src
end                                                 #src

using GLMakie
using IncompressibleNavierStokes
using JLD2
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Random
using Zygote

# Floating point precision
T = Float32

# # To use CPU: Do not move any arrays
# device = identity

# To use GPU, use `cu` to move arrays to the GPU.
# Note: `cu` converts to Float32
using CUDA
using LuxCUDA
device = cu

# Viscosity model
Re = T(2_000)
viscosity_model = LaminarModel(; Re)

# A 2D grid is a Cartesian product of two vectors
s = 4
n_les = 128
n_dns = s * n_les

lims = T(0), T(1)
x_dns = LinRange(lims..., n_dns + 1)
y_dns = LinRange(lims..., n_dns + 1)
x_les = x_dns[1:s:end]
y_les = y_dns[1:s:end]

# Build setup and assemble operators
dns = Setup(x_dns, y_dns; viscosity_model);
les = Setup(x_les, y_les; viscosity_model);

# Filter
(; KV, Kp) = operator_filter(dns.grid, dns.boundary_conditions, s);

# Since the grid is uniform and identical for x and y, we may use a specialized
# spectral pressure solver
pressure_solver = SpectralPressureSolver(dns);
pressure_solver_les = SpectralPressureSolver(les);

filter_saver(setup, KV, Kp, Ωbar; nupdate = 1, bc_vectors = get_bc_vectors(setup, T(0))) =
    processor(
        function (step_observer)
            (; Ω) = setup.grid
            KVmom = Ωbar * KV * Diagonal(1 ./ Ω)
            T = eltype(setup.grid.x)
            _V = fill(zeros(T, 0), 0)
            _F = fill(zeros(T, 0), 0)
            _FG = fill(zeros(T, 0), 0)
            _p = fill(zeros(T, 0), 0)
            _t = fill(zero(T), 0)
            on(step_observer) do (; V, p, t)
                F, = momentum(V, V, p, t, setup; bc_vectors, nopressure = true)
                FG, = momentum(V, V, p, t, setup; bc_vectors, nopressure = false)
                push!(_V, KV * Array(V))
                push!(_F, KVmom * Array(F))
                push!(_FG, KVmom * Array(FG))
                push!(_p, Kp * Array(p))
                push!(_t, t)
            end
            (; V = _V, F = _F, FG = _FG, p = _p, t = _t)
        end;
        nupdate,
    )

# Time interval, including burn-in time
t_start, t_burn, t_end = T(0.0), T(0.1), T(0.2)

# Number of time steps
Δt = T(2e-4)
n_t = round(Int, (t_end - t_burn) / Δt)

# Number of random initial conditions
n_ic = 10

# Filtered quantities to store
filtered = (;
    V = zeros(T, n_les * n_les * 2, n_t + 1, n_ic),
    F = zeros(T, n_les * n_les * 2, n_t + 1, n_ic),
    FG = zeros(T, n_les * n_les * 2, n_t + 1, n_ic),
    p = zeros(T, n_les * n_les, n_t + 1, n_ic),
)

Base.summarysize(filtered) / 1e6

# Iteration processors
processors = (
    filter_saver(
        device(dns),
        KV,
        Kp,
        les.grid.Ω;
        bc_vectors = device(get_bc_vectors(dns, t_start)),
    ),
    step_logger(; nupdate = 10),
);

for i_ic = 1:n_ic
    @info "Generating data for IC $i_ic of $n_ic"

    # Initial conditions
    V₀, p₀ = random_field(dns; A = T(10_000_000), σ = T(30), s = 5, pressure_solver)

    # Solve burn-in DNS
    @info "Burn-in for IC $i_ic of $n_ic"
    V, p, outputs = solve_unsteady(
        dns,
        V₀,
        p₀,
        (t_start, t_burn);
        Δt = T(2e-4),
        processors = (step_logger(; nupdate = 10),),
        pressure_solver,
        inplace = true,
        device,
    )

    # Solve DNS and store filtered quantities
    @info "Solving DNS for IC $i_ic of $n_ic"
    V, p, outputs = solve_unsteady(
        dns,
        V,
        p,
        (t_burn, t_end);
        Δt = T(2e-4),
        processors,
        pressure_solver,
        inplace = true,
        device,
    )
    f = outputs[1]

    # Store result for current IC
    filtered.V[:, :, i_ic] = stack(f.V)
    filtered.F[:, :, i_ic] = stack(f.F)
    filtered.FG[:, :, i_ic] = stack(f.FG)
    filtered.p[:, :, i_ic] = stack(f.p)
end

# jldsave("output/filtered/filtered.jld2"; filtered)
filtered = load("output/filtered/filtered.jld2", "filtered")

size(filtered.V)

plot_vorticity(les, filtered.V[:, end, 1], t_burn)

# Compute commutator errors
bc_vectors = get_bc_vectors(les, t_burn)
commutator_error = zero(filtered.F)
pbar = filtered.p[:, 1, 1]
for i_t = 1:n_t, i_ic = 1:n_ic
    @info "Computing commutator error for time $i_t of $n_t, IC $i_ic of $n_ic"
    V = filtered.V[:, i_t, i_ic]
    F = filtered.F[:, i_t, i_ic]
    Fbar, = momentum(V, V, pbar, t_burn, les; bc_vectors, nopressure = true)
    commutator_error[:, i_t, i_ic] .= F .- Fbar
end

norm(commutator_error[:, 1, 1]) / norm(filtered.F[:, 1, 1])

# closure, θ₀ = cnn(
#     les,
#     [5, 5, 5],
#     [2, 8, 8, 2],
#     [leakyrelu, leakyrelu, identity],
#     [true, true, false];
# )

closure, θ₀ = fno(
    # Setup
    les,

    # Cut-off wavenumbers
    [8, 8, 8],

    # Channel sizes
    [16, 8, 8],

    # Fourier activations
    [gelu, gelu, identity],

    # Dense activation
    gelu,
)

length(θ₀)

loss(x, y, θ) = sum(abs2, closure(x, θ) - y) / sum(abs2, y)

loss(filtered.V[:, 1, 1], commutator_error[:, 1, 1], θ₀)

function create_loss(x, y; nuse = size(x, 2), device = identity)
    x = reshape(x, size(x, 1), :)
    y = reshape(y, size(y, 1), :)
    nsample = size(x, 2)
    d = ndims(x)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        xuse = Zygote.@ignore device(Array(selectdim(x, d, i)))
        yuse = Zygote.@ignore device(Array(selectdim(y, d, i)))
        loss(xuse, yuse, θ)
    end
end

randloss = create_loss(filtered.V, commutator_error; nuse = 50, device)

θ = 5.0f-2 * device(θ₀)

randloss(θ)

@time first(gradient(randloss, θ));

V_test = device(reshape(filtered.V[:, 1:20, 1:2], :, 40))
c_test = device(reshape(commutator_error[:, 1:20, 1:2], :, 40))

opt = Optimisers.setup(Adam(1.0f-2), θ)

obs = Observable([(0, T(0))])

fig = lines(obs; axis = (; title = "Relative prediction error", xlabel = "Iteration"))
hlines!([1.0f0])
display(fig)

obs[] = fill((0, T(0)), 0)
j = 0

nplot = 10
niter = 500
for i = 1:niter
    g = first(gradient(randloss, θ))
    opt, θ = Optimisers.update(opt, θ, g)
    if i % nplot == 0
        e_test = norm(closure(V_test, θ) - c_test) / norm(c_test)
        @info "Iteration $i\trelative test error: $e_test"
        _i = (j += nplot)
        obs[] = push!(obs[], (_i, e_test))
        autolimits!(fig.axis)
    end
end

# jldsave("output/theta.jld2"; θ = Array(θ))
# θθ = load("output/theta.jld2")
# θθ = θθ["θ"]
# θθ = cu(θθ)
# θ .= θθ

relative_error(closure(V_test, θ), c_test)

size(filtered.V)

devles = device(les);

V_nm, p_nm, outputs_nm = solve_unsteady(
    les,
    filtered.V[:, 1, 1],
    filtered.p[:, 1, 1],
    (t_burn, t_end);
    Δt = T(2e-4),
    processors = (
        step_logger(; nupdate = 10),
        field_plotter(devles; type = heatmap, nupdate = 1),
    ),
    pressure_solver = pressure_solver_les,
    inplace = false,
    device,
)

V_fno, p_fno, outputs_fno = solve_unsteady(
    (; les..., closure_model = V -> closure(V, θ)),
    filtered.V[:, 1, 1],
    filtered.p[:, 1, 1],
    (t_burn, t_end);
    Δt = T(2e-4),
    processors = (
        step_logger(; nupdate = 10),
        field_plotter(devles; type = heatmap, nupdate = 1),
    ),
    pressure_solver = pressure_solver_les,
    inplace = false,
    device,
)

V = filtered.V[:, end, 1]
p = filtered.p[:, end, 1]

relative_error(V_nm, V)
relative_error(V_fno, V)

plot_vorticity(les, V_nm, t_end)
plot_vorticity(les, V_fno, t_end)
plot_vorticity(les, V, t_end)

CUDA.memory_status()
GC.gc()
CUDA.reclaim()
