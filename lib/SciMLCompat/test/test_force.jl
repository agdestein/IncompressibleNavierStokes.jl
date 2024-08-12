using SciMLCompat
using IncompressibleNavierStokes
using OrdinaryDiffEq
using ComponentArrays

TIME_TOL = 1.1

# Setup
T = Float32
Re = T(1_000)
dt = T(1e-3)
saveat = T(1e-2)
n = 32
N = n + 2
trange = T(0), T(1)
x = LinRange(trange..., n + 1), LinRange(trange..., n + 1);
setup = Setup(x...; Re);
ustart = random_field(setup, 0.0);
psolver = psolver_direct(setup);

u0 = stack(ustart)
f = create_right_hand_side(setup, psolver)
op_t_1 = f(u0, nothing, 0.0)

# Solve the ODE using SciML
prob = ODEProblem(f, u0, trange)
sol, time_op, allocation_op, gc_op, memory_counters_op = @timed solve(
    prob,
    Tsit5();
    dt = dt,
    saveat = saveat
);

f_ip = create_right_hand_side_inplace(setup, psolver)
du = similar(u0)
f_ip(du, u0, nothing, 0.0);

# Solve the ODE using SciML
prob_ip = ODEProblem(f, u0, trange);
sol_ip, time_ip, allocation_ip, gc_ip, memory_counters_ip = @timed solve(
    prob_ip,
    Tsit5();
    dt = dt,
    saveat = saveat
);

f_e = create_right_hand_side_enzyme(get_backend(u0), setup, T, n)
f_e(du, u0, nothing, 0.0)

prob_e = ODEProblem(f, u0, trange);
sol_e, time_e, allocation_e, gc_e, memory_counters_e = @timed solve(
    prob_e,
    Tsit5();
    dt = dt,
    saveat = saveat
);

# test the solutions 
@assert sol.u[end] ≈ sol_ip.u[end] ≈ sol_e.u[end]
# Test the time
@assert time_e < time_op "enzyme force too slow: time_enzyme = $time_e, time_out_of_place = $time_op"
@assert time_e < TIME_TOL * time_ip "enzyme force too slow: time_enzyme = $time_e, time_in_place = $time_ip"
# Test the memory allocation
@assert allocation_e < allocation_op "enzyme force uses too much memory: allocation_enzyme = $allocation_e, allocation_out_of_place = $allocation_op"
@assert allocation_e < TIME_TOL * allocation_ip "enzyme force uses too much memory: allocation_enzyme = $allocation_e, allocation_in_place = $allocation_ip"
