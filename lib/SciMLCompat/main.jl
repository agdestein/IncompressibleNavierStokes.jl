using SciMLCompat
using IncompressibleNavierStokes
using OrdinaryDiffEq
using KernelAbstractions
using ComponentArrays
using GLMakie

# Setup
T = Float32
Re = T(2e3)
n = 128
N = n + 2
trange = T(0), T(1)
x = LinRange(trange..., n + 1), LinRange(trange..., n + 1);
setup = Setup(x...; Re);
ustart = random_field(setup, 0.0);
psolver = psolver_direct(setup);

# SciML-compatible right hand side function
# Note: Requires `stack(u)` to create one array
u0 = stack(ustart)
f = create_right_hand_side(setup, psolver)
op_t_1 = f(u0, nothing, 0.0)

# Solve the ODE using SciML
prob = ODEProblem(f, u0, trange)
sol, usedtime, allocation, gc, memory_counters = @timed solve(
    prob,
    Tsit5();
    # adaptive = false,
    dt = 1e-3,
);

f_ip = create_right_hand_side_inplace(setup, psolver)
du = similar(u0)
f_ip(du, u0, nothing, 0.0);

# Solve the ODE using SciML
prob = ODEProblem(f, u0, trange);
sol_ip, usedtime_ip, allocation_ip, gc_ip, memory_counters_ip = @timed solve(
    prob,
    Tsit5();
    # adaptive = false,
    dt = 1e-3,
);

f_e = create_right_hand_side_enzyme(get_backend(u0), setup, T, n)
P = ComponentArray(;
    f = zeros(T, (N, N, 2)),
    div = zeros(T, (N, N)),
    p = zeros(T, (N, N)),
    ft = zeros(T, n * n + 1),
    pt = zeros(T, n * n + 1),
)
f_e(du, u0, nothing, 0.0)

prob = ODEProblem(f, u0, trange);
sol_e, usedtime_e, allocation_e, gc_e, memory_counters_e = @timed solve(
    prob,
    Tsit5();
    # adaptive = false,
    dt = 1e-3,
    #    p = P,
);

# Now plot the times for comparison
using Plots
p1 = bar(
    ["SciML", "SciML inplace", "Enzyme"],
    [usedtime, usedtime_ip, usedtime_e];
    title = "Time comparison",
    ylabel = "Time (s)",
    legend = false,
)
p2 = bar(
    ["SciML", "SciML inplace", "Enzyme"],
    [allocation, allocation_ip, allocation_e];
    title = "Allocation comparison",
    ylabel = "Memory (MB)",
    legend = false,
)
p3 = bar(
    ["SciML", "SciML inplace", "Enzyme"],
    [gc, gc_ip, gc_e];
    title = "GC comparison",
    ylabel = "GC (s)",
    legend = false,
)
Plots.plot(p1, p2, p3; layout = (3, 1), size = (800, 800))

# and compare the solutions
p1 = Plots.heatmap(sol.u[end][:, :, 1]; title = "SciML")
p2 = Plots.heatmap(sol_ip.u[end][:, :, 1]; title = "SciML inplace")
p3 = Plots.heatmap(sol_e.u[end][:, :, 1]; title = "Enzyme")
Plots.plot(p1, p2, p3; layout = (1, 3), size = (800, 400))

@assert sol.u[end] ≈ sol_ip.u[end] ≈ sol_e.u[end]

Plots.heatmap(sol.u[end][:, :, 1] - sol_e.u[end][:, :, 1])
Plots.heatmap(sol.u[end][:, :, 1] - sol_ip.u[end][:, :, 1])
