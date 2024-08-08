using CairoMakie
using IncompressibleNavierStokes
INS = IncompressibleNavierStokes


# Setup and initial condition
T = Float32
ArrayType = Array
Re = T(1_000)
n = 128
n = 64
n = 32
N = n+2
# this is the size of the domain, do not mix it with the time
lims = T(0), T(1);
x , y = LinRange(lims..., n + 1), LinRange(lims..., n + 1);
setup = INS.Setup(x, y; Re, ArrayType);
ustart = INS.random_field(setup, T(0));
psolver = INS.psolver_direct(setup);
dt = T(1e-3);
tfinal = T(0.2)
ndt = ceil(Int,tfinal/dt)
trange = [T(0), tfinal];
savevery = 20;
saveat = savevery * dt;
npoints = ceil(Int, ndt/savevery)


# Solving using INS semi-implicit method
(state, outputs), time_ins, allocation, gc, memory_counters = @timed INS.solve_unsteady(;
    setup,
    ustart,
    tlims = trange,
    Δt = dt,
    psolver = psolver,
);

all_INS_states = []
push!(all_INS_states, ustart)
for i in 1:npoints
    oldstate = all_INS_states[end]
    thisstate, outputs = INS.solve_unsteady(;
        setup,
        ustart = oldstate,
        tlims = [T(0), dt*savevery],
        Δt = dt,
        psolver = psolver,
    );
    push!(all_INS_states, thisstate.u)
end
all_INS_states
@assert all_INS_states[end-1] != state.u
@assert all_INS_states[end] == state.u


############# Using SciML
using DifferentialEquations

# Projected force for SciML, to use in CNODE
F = similar(stack(ustart));
# and prepare a cache for the force
cache_F = (F[:,:,1], F[:,:,2]);
cache_div = INS.divergence(ustart,setup);
cache_p = INS.pressure(ustart, nothing, 0.0f0, setup; psolver);
Ω = setup.grid.Ω;


# Get the cache for the poisson solver
include("./INS_SciMLinterface.jl")
cache_ftemp, cache_ptemp, fact, cache_viewrange, cache_Ip = my_cache_psolver(setup.grid.x[1], setup)
# and use it to precompile an Enzyme-compatible psolver
my_psolve! = generate_psolver(cache_viewrange, cache_Ip, fact)
# In a similar way, get the function for the divergence 
mydivergence! = get_divergence!(cache_p, setup);
# and the function to apply the pressure
myapplypressure! = get_applypressure!(ustart, setup);
# and the momentum
my_momentum! = get_momentum!(cache_F, ustart, nothing, setup);
# and the boundary conditions
my_bc_p! = get_bc_p!(cache_p, setup);
my_bc_u! = get_bc_u!(cache_F, setup);



# Define the cache for the force 
using ComponentArrays
using KernelAbstractions
# I have also to take the grid size to stack into P
(; grid) = setup;
(; Δ, Δu, A, Ω) = grid;
# Watch out for the type of this
P = ComponentArray(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), temp=zeros(T,(n+2,n+2)))
P = ComponentArray(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)))
@assert eltype(P)==T


# **********************8
# * Force in place
F_ip(du, u, p, t) = begin
    u_view = eachslice(u; dims = 3)
    F = eachslice(p.f; dims = 3)
    my_bc_u!(u_view)
    my_momentum!(F, u_view, t )
    my_bc_u!(F)
    mydivergence!(p.div, F, p.p)
    @. p.div *= Ω
    my_psolve!(p.p, p.div, p.ft, p.pt)
    my_bc_p!(p.p)
    myapplypressure!(F, p.p)
    my_bc_u!(F)
    du[:,:,1] .= F[1]
    du[:,:,2] .= F[2]
    nothing
end;
temp = similar(stack(ustart));
F_ip(temp, stack(ustart), P, 0.0f0)


# Solve the ODE using ODEProblem
prob = ODEProblem{true}(F_ip, stack(ustart), trange, p=P)
sol_ode, time_ode, allocation_ode, gc_ode, memory_counters_ode = @timed solve(
    prob,
    p = P,
    RK4();
    dt = dt,
    saveat = saveat,
);



# ------ Use Lux to create a dummy_NN
import Random, Lux;
Random.seed!(123);
rng = Random.default_rng();
#dummy_NN = Lux.Chain(
#    Lux.ReshapeLayer((N,N,1)),
#    Lux.Conv((3, 3), 1 => 1, pad=(1, 1)),
#    x -> view(x, :),  # Flatten the output
#)
dummy_NN = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2, pad=(1, 1), stride=(1, 1)),
    x -> view(x, :),  
)
θ0, st0 = Lux.setup(rng, dummy_NN)
st_node = st0

using ComponentArrays
θ_node = ComponentArray(θ0)
Lux.apply(dummy_NN, stack(ustart), θ_node, st0)[1];

P = ComponentArray(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), θ=copy(θ_node))
@assert eltype(P)==T
Lux.apply(dummy_NN, stack(ustart), P.θ, st0)[1];


# Force+NN in-place version
dudt_nn(du, u, P, t) = begin 
    F_ip(du, u, P, t) 
    view(du, :) .= view(du, :) .+ Lux.apply(dummy_NN, u, P.θ , st_node)[1]
    nothing
end


temp = similar(stack(ustart));
dudt_nn(temp, stack(ustart), P, 0.0f0)
prob_node = ODEProblem{true}(dudt_nn, stack(ustart), trange, p=P);

u0stacked = stack(ustart);
sol_node, time_node, allocation_node, gc_node, memory_counters_node = @timed solve(prob_node, RK4(), u0 = u0stacked, p = P, saveat = saveat, dt=dt);


println("Done run")

# Compare the times of the different methods via a bar plot
using Plots
p1=Plots.bar(["INS", "ODE", "CNODE"], [time_ins, time_ode, time_node], xlabel = "Method", ylabel = "Time (s)", title = "Time comparison")
# Compare the memory allocation
p2=Plots.bar(["INS", "ODE", "CNODE"], [memory_counters.allocd, memory_counters_ode.allocd, memory_counters_node.allocd], xlabel = "Method", ylabel = "Memory (bytes)", title = "Memory comparison")
# Compare the number of garbage collections
p3=Plots.bar(["INS", "ODE", "CNODE"], [gc, gc_ode, gc_node], xlabel = "Method", ylabel = "Number of GC", title = "GC comparison")

Plots.plot(p1, p2, p3, layout=(3,1), size=(600, 800))


# Plot the final state
using Plots
p1=Plots.heatmap(title="u in SciML ODE",sol_ode.u[end][:, :, 1])
p2=Plots.heatmap(title="u in SciML CNODE",sol_node.u[end][:, :, 1])
p3=Plots.heatmap(title="u in INS",state.u[1])
# and compare them
p4=Plots.heatmap(title="u_INS-u_ODE",state.u[1] - sol_ode.u[end][:, :, 1])
p5=Plots.heatmap(title="u_INS-u_CNODE",state.u[1] - sol_node.u[end][:, :, 1])
p6=Plots.heatmap(title="u_CNODE-u_ODE",sol_node.u[end][:, :, 1] - sol_ode.u[end][:, :, 1])
Plots.plot(p1, p2, p3, p4,p5,p6, layout=(2,3), size=(900,600))


# Compute the divergence of the final state
div_INS = INS.divergence(state.u, setup);
div_ode = INS.divergence((sol_ode.u[end][:,:,1],sol_ode.u[end][:,:,2]), setup);
div_node = INS.divergence((sol_node.u[end][:,:,1],sol_node.u[end][:,:,2]), setup);
p1 = Plots.heatmap(title="div_INS",div_INS)
p2 = Plots.heatmap(title="div_ODE",div_ode)
p3 = Plots.heatmap(title="div_NODE",div_node)
Plots.plot(p1, p2, p3, layout=(1,3), size=(900,300))



########################
# Test the autodiff using Enzyme 
using Enzyme
using ComponentArrays
using SciMLSensitivity


# First test Enzyme for something that does not make sense bu it has the structure of a priori loss
U = stack(state.u);
function fen(u0, p, temp)
    dudt_nn(temp, u0, p, 0.0f0)
    return sum(U - temp)
end
u0stacked = stack(ustart);
du = Enzyme.make_zero(u0stacked);
dP = Enzyme.make_zero(P);
temp = similar(stack(ustart));
dtemp = Enzyme.make_zero(temp);
# Compute the autodiff using Enzyme
@timed Enzyme.autodiff(Enzyme.Reverse, fen, Active, DuplicatedNoNeed(u0stacked, du), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(temp, dtemp))
# the gradient that we need is only the following
dP.θ
# this shows us that Enzyme can differentiate our force. But what about SciML solvers?
println("Tested a priori")


# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
# and remember to set a small dt
dt = T(1e-3);
trange = [T(0), T(2e-3)]
saveat = [dt, 2dt];
u0stacked = stack(ustart);
P = ComponentArray(f=zeros(T, (n+2,n+2,2)),div=zeros(T,(n+2,n+2)), p=zeros(T,(n+2,n+2)), ft=zeros(T,size(cache_ftemp)), pt=zeros(T,size(cache_ptemp)), θ=copy(θ_node))
prob = ODEProblem{true}(dudt_nn, u0stacked, trange, p=P)
ode_data = Array(solve(prob, RK4(), u0 = u0stacked, p = P, saveat = saveat))
ode_data += T(0.1)*rand(Float32, size(ode_data))


# the loss has to be in place 
function loss(l::Vector{Float32},P, u0::Array{Float32}, tspan::Vector{Float32}, t::Vector{Float32})
    myprob = ODEProblem{true}(dudt_nn, u0, tspan, P)
    pred = Array(solve(myprob, RK4(), u0 = u0, p = P, saveat=t))
    l .= Float32(sum(abs2, ode_data- pred))
    nothing
end
l=[T(0.0)];
loss(l,P, u0stacked, trange, saveat);
l


# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [T(0.0)];
dl = Enzyme.make_zero(l) .+T(1);
dP = Enzyme.make_zero(P);
du = Enzyme.make_zero(u0stacked);
@timed Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed(l, dl), DuplicatedNoNeed(P, dP), DuplicatedNoNeed(u0stacked, du), Const(trange), Const(saveat))
dP.θ
    




println("Now defining the gradient function")
extra_par = [u0stacked, trange, saveat, du, dP, P];
Textra = typeof(extra_par);
function loss_gradient(G, extra_par) 
    u0, trange, saveat, du0, dP, P = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dP)
    # And remember to pass the seed to the loss funciton with the dual part set to 1
    Enzyme.autodiff(Enzyme.Reverse, loss, DuplicatedNoNeed([T(0)], [T(1)]), DuplicatedNoNeed(P,dP), DuplicatedNoNeed(u0, du0), Const(trange), Const(saveat))
    # The gradient matters only for theta
    G .= dP.θ
    nothing
end

# Trigger the gradient
G = copy(dP.θ);
oo = loss_gradient(G, extra_par)


# This is to call loss using only P
function over_loss(θ, p)
    # Here we are updating P.θ in place
    p.θ .= θ
    loss(l,p, u0stacked, trange, saveat);
    return l
end
callback = function (θ,l; doplot = false)
    println(l)
    return false
end
callback(P, over_loss(P.θ, P))


using SciMLSensitivity, Optimization, OptimizationOptimisers, Optimisers
optf = Optimization.OptimizationFunction((p,u)->over_loss(p,u[end]), grad=(G,p,e)->loss_gradient(G,e))
optprob = Optimization.OptimizationProblem(optf, P.θ, extra_par)


result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100)

