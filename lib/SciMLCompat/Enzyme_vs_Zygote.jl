using CairoMakie
using Optimization
using OptimizationOptimisers
using Optimisers
using Lux
using Random
Random.seed!(123);
rng = Random.default_rng();
using ComponentArrays
using KernelAbstractions
using SparseArrays
using Plots
using Enzyme
Enzyme.API.runtimeActivity!(true)
using Zygote
using IncompressibleNavierStokes
using SciMLCompat
INS = IncompressibleNavierStokes

# Define the problem
T = Float32
ArrayType = Array
Re = T(1_000)
lims = T(0), T(1);
dt = T(1e-3);
trange = [T(0), T(1)]
saveat = [i*dt for i in 1:div(1, dt)]
_backend = get_backend(rand(Float32, 10))

# and a dns and les grid 
n_dns = 128
N_dns = n_dns + 2
n_les = 64
N_les = n_les + 2
x_dns, y_dns = LinRange(lims..., n_dns + 1), LinRange(lims..., n_dns + 1);
x_les, y_les = LinRange(lims..., n_les + 1), LinRange(lims..., n_les + 1);
setup_dns = INS.Setup(x_dns, y_dns; Re, ArrayType);
setup_les = INS.Setup(x_les, y_les; Re, ArrayType);

# ### Filter
function create_filter_matrix_2d(dx_dns, dx_les, N_dns, N_les, ΔΦ, kernel_type, MY_TYPE=Float64)
    ## Filter kernels
    gaussian(Δ, x, y) = MY_TYPE(sqrt(6 / π) / Δ * exp(-6 * (x^2 + y^2) / Δ^2))
    top_hat(Δ, x, y) =  MY_TYPE((abs(x) ≤ Δ / 2) * (abs(y) ≤ Δ / 2) / (Δ^2))

    ## Choose kernel
    kernel = kernel_type == "gaussian" ? gaussian : top_hat

    x_dns = collect(0:dx_dns:((N_dns - 1) * dx_dns))
    y_dns = collect(0:dx_dns:((N_dns - 1) * dx_dns))
    x_les = collect(0:dx_les:((N_les - 1) * dx_les))
    y_les = collect(0:dx_les:((N_les - 1) * dx_les))

    ## Discrete filter matrix (with periodic extension and threshold for sparsity)
    Φ = sum(-1:1) do z_x
        sum(-1:1) do z_y
            d_x = @. x_les - x_dns' - z_x
            d_y = @. y_les - y_dns' - z_y
            if kernel_type == "gaussian"
                @. kernel(ΔΦ, d_x, d_y) * (abs(d_x) ≤ 3 / 2 * ΔΦ) * (abs(d_y) ≤ 3 / 2 * ΔΦ)
            else
                @. kernel(ΔΦ, d_x, d_y)
            end
        end
    end
    Φ = Φ ./ sum(Φ; dims = 2) ## Normalize weights
    Φ = sparse(Φ)
    dropzeros!(Φ)
    return Φ
end
dx_dns = x_dns[2] - x_dns[1]
dx_les = x_les[2] - x_les[1] 
# To get the LES, we use a Gaussian filter kernel, truncated to zero outside of $3 / 2$ filter widths.
ΔΦ = 5 * dx_les
Φ = create_filter_matrix_2d(dx_dns, dx_les, N_dns, N_les, ΔΦ, "gaussian", T)
function apply_filter(ϕ, u)
    vx = ϕ * u[1] * ϕ'
    vy = ϕ * u[2] * ϕ'
    return (vx, vy)
end


# Initial condition
u0_dns = INS.random_field(setup_dns, T(0));
@assert u0_dns[1][1,:] == u0_dns[1][end-1,:]
@assert u0_dns[1][end,:] == u0_dns[1][2,:]
#u0_les = apply_filter(Φ, (u0_dns[1][2:end-1, 2:end-1], u0_dns[2][2:end-1, 2:end-1])) 
u0_les = apply_filter(Φ, u0_dns) 
@assert u0_les[1][1,:] == u0_les[1][end-1,:]
@assert u0_les[1][end,:] == u0_les[1][2,:]
p1 = Plots.heatmap(u0_dns[1], title = "DNS initial condition")
p2 = Plots.heatmap(u0_les[1], title = "LES initial condition")
Plots.plot(p1, p2, layout=(1,2), size=(800, 400))

# Get the forces for NS
# Zygote force (out-of-place)
F_out_dns = create_right_hand_side(setup_dns, INS.psolver_direct(setup_dns))
F_out_les = create_right_hand_side(setup_les, INS.psolver_direct(setup_les))
# and in-place version
F_in_dns = create_right_hand_side_enzyme(_backend, setup_dns, T, n_dns)
F_in_les = create_right_hand_side_enzyme(_backend, setup_les, T, n_les)

# define sciml problems 
prob_dns = ODEProblem{true}(F_in_dns, stack(u0_dns), trange)
prob_les = ODEProblem{true}(F_in_les, stack(u0_les), trange)

# Solve the exact solutions
using DifferentialEquations
sol_dns, time_dns, allocation_dns, gc_dns, memory_counters_dns =     @timed solve(prob_dns, Tsit5(); dt = dt, saveat = saveat);
sol_les, time_les, allocation_les, gc_les, memory_counters_les =     @timed solve(prob_les, Tsit5(); dt = dt, saveat = saveat);

# Compare the times of the different methods via a bar plot
p1=Plots.bar(["DNS", "LES"], [time_dns, time_les], xlabel = "Method", ylabel = "Time (s)", title = "Time comparison")
# Compare the memory allocation
p2=Plots.bar(["DNS", "LES"], [memory_counters_dns.allocd, memory_counters_les.allocd], xlabel = "Method", ylabel = "Memory (bytes)", title = "Memory comparison")
# Compare the number of garbage collections
p3=Plots.bar(["DNS", "LES"], [gc_dns, gc_les], xlabel = "Method", ylabel = "Number of GC", title = "GC comparison")
Plots.plot(p1, p2, p3, layout=(3,1), size=(600, 800))

# and show an animation of the solution
anim = Animation()
fig = Plots.plot(layout = (3, 1), size = (300, 800))
@gif for i in 1:10:length(saveat)
    p1 = Plots.heatmap(sol_dns.u[i][:, :, 1], title = "DNS")
    p2 = Plots.heatmap(sol_les.u[i][:, :, 1], title = "LES")
    uu = (sol_dns.u[i][:,:,1], sol_dns.u[i][:,:,2])
    u_filtered = apply_filter(Φ, uu)
    p3 = Plots.heatmap(u_filtered[1], title = "Filtered DNS")
    title = "Time: $(round((i - 1) * dt, digits = 2))"
    fig = Plots.plot(p1, p2, p3, size=(300,800), layout = (3, 1), suptitle = title)
    frame(anim, fig)
end


#####################
# A priori training
#####################
# The target is the filtered dns force
all_F_filtered = []
for i in 1:length(saveat)
    F_dns = F_out_dns(sol_dns.u[i], nothing, 0)
    F_filt = apply_filter(Φ, (F_dns[:,:,1],F_dns[:,:,2]))
    push!(all_F_filtered, stack(F_filt))
end

# define a dummy_NN to train
dummy_NN_Z = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
)
dummy_NN_E = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    x -> view(x, :),
)
θ_Z, st_Z = Lux.setup(rng, dummy_NN_Z)
θ_E, st_E = Lux.setup(rng, dummy_NN_E)
θ_Z = ComponentArray(θ_Z)
θ_E = ComponentArray(θ_E)
# set same initial conditions for the dummy NN
θ_E.layer_2 = θ_Z.layer_2
θ_E.layer_3 = θ_Z.layer_3
θ_E.layer_4 = θ_Z.layer_4
@assert view(Lux.apply(dummy_NN_Z, stack(u0_les), θ_Z, st_Z)[1],:) == Lux.apply(dummy_NN_E, stack(u0_les), θ_E, st_E)[1]

# Define the right hand side function with the neural network closure   
dudt_nn_Z(u, θ, t) = begin
    F_out_les(u, θ, t) .+ Lux.apply(dummy_NN_Z, u, θ, st_Z)[1][:,:,:,1]
end
dudt_nn_Z(stack(u0_les), θ_Z, T(0))

# define the loss function
npoints = 32
function loss_priori_Z(p)
    l = 0
    d = 0
    # select a random set of points 
    for i in 1:npoints
        i = @Zygote.ignore rand(1:length(saveat))
        l += sum(abs2, all_F_filtered[i] .- dudt_nn_Z(sol_les.u[i], p, 0.0))
        d += sum(abs2, all_F_filtered[i])
    end
    return l/d
end

# and train using Zygote
callback = function (θ, l; doplot = false)
    println(l)
    return false
end
callback(θ_Z, loss_priori_Z(θ_Z))

optf = Optimization.OptimizationFunction((x,p)->loss_priori_Z(x), Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ_Z)

result_priori_Z, time_priori_Z, alloc_priori_Z, gc_priori_Z, mem_priori_Z = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1),
    callback = callback,
    maxiters = 50,
)
θ_priori_Z = result_priori_Z.u

# Now the same (a priori) but using Enzyme
dudt_nn_E(du, u, θ, t) = begin
    F_in_les(du, u, nothing, t)
    view(du, :) .= view(du, :) .+ Lux.apply(dummy_NN_E, u, θ, st_E)[1]
    nothing
end
temp = similar(stack(u0_les));
dudt_nn_E(temp, stack(u0_les), θ_E, 0.0)


function loss_priori_E(l, u, p, temp, t)
    lnum = 0
    lden = 0
    # select a random set of points 
    for i in 1:npoints
        i = rand(1:length(saveat))
        dudt_nn_E(temp, u[i], p, 0.0)
        lnum += sum(abs2, all_F_filtered[i] .- temp)
        lden += sum(abs2, all_F_filtered[i])
    end
    l .= T(lnum/lden)
    nothing
end
u = u_ini = sol_les.u 
du = Enzyme.make_zero(u);
dθ = Enzyme.make_zero(θ_E);
dtemp = Enzyme.make_zero(temp);
l = [T(0)]
loss_priori_E(l, u, θ_E, temp, 0)
l

Enzyme.autodiff(
    Enzyme.Reverse,
    loss_priori_E,
    DuplicatedNoNeed([T(0)], [T(1)]),
    DuplicatedNoNeed(u, du),
    DuplicatedNoNeed(θ_E, dθ),
    DuplicatedNoNeed(temp, dtemp),
    Const(0),
)

# For enzyme, we have to define the gradient function
extra_par = [u_ini, du, temp, dtemp, dθ];
function loss_gradient(G, θ, extra_par)
    u, du, temp, dtemp, dθ = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dθ)
    # And remember to pass the seed to the loss function with the dual part set to 1
    Enzyme.autodiff(
        Enzyme.Reverse,
        loss_priori_E,
        DuplicatedNoNeed([T(0)], [T(1)]),
        DuplicatedNoNeed(u, du),
        DuplicatedNoNeed(θ, dθ),
        DuplicatedNoNeed(temp, dtemp),
        Const(0),
    )
    # The gradient matters only for theta
    G .= dθ
    nothing
end

# Trigger the gradient
G = copy(dθ);
loss_gradient(G, θ_E, extra_par)
G

function wrapped_loss(p)
    loss_priori_E(l,u_ini,p,temp,0.0)
    l[1]
end
wrapped_loss(θ_E)

optf = Optimization.OptimizationFunction(
    (u, _) -> wrapped_loss(u),
    grad = (G, u, p) -> loss_gradient(G, u, p),
    Optimization.AutoEnzyme()
)
optprob = Optimization.OptimizationProblem(optf, θ_E, extra_par)

result_priori_E, time_priori_E, alloc_priori_E, gc_priori_E, mem_priori_E = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1),
    callback = callback,
    maxiters = 50,
)
θ_priori_E = result_priori_E.u
@assert result_priori_E.u != θ_E

# Compare the times of the different methods via a bar plot
p1=Plots.bar(["Zygote", "Enzyme"], [time_priori_Z, time_priori_E], xlabel = "Method", ylabel = "Time (s)", title = "Time comparison")
p2=Plots.bar(["Zygote", "Enzyme"], [mem_priori_Z.allocd, mem_priori_E.allocd], xlabel = "Method", ylabel = "Memory (bytes)", title = "Memory comparison")
p3=Plots.bar(["Zygote", "Enzyme"], [gc_priori_Z, gc_priori_E], xlabel = "Method", ylabel = "Number of GC", title = "GC comparison")
Plots.plot(p1, p2, p3, layout=(3,1), size=(600, 800))
# and compare the total_loss at the end of the training
function total_loss_Z(p)
    l = 0
    d = 0
    for i in 1:length(saveat)
        l += sum(abs2, all_F_filtered[i] .- dudt_nn_Z(sol_les.u[i], p, 0.0))
        d += sum(abs2, all_F_filtered[i])
    end
    return l/d
end
tl0_Z = total_loss_Z(θ_Z)
tl_Z = total_loss_Z(θ_priori_Z)
function total_loss_E(p)
    l = 0
    d = 0
    for i in 1:length(saveat)
        dudt_nn_E(temp,sol_les.u[i], p, 0.0)
        l += sum(abs2, all_F_filtered[i] .- temp)
        d += sum(abs2, all_F_filtered[i])
    end
    return l/d
end
tl0_E = total_loss_E(θ_E)
@assert tl0_Z == tl0_E
tl_E = total_loss_E(θ_priori_E)

# Compare the total loss
Plots.bar(["(Random)", "Zygote", "Enzyme"], [tl0_E,tl_Z, tl_E], xlabel = "Method", ylabel = "Total Loss", title = "Total Loss comparison", yscale=:log10)

########################
# A posteriori
########################
using SciMLSensitivity
# define the loss function
nunroll = 5
saveat_loss = [i*dt for i in 1:nunroll]
tspan = [T(0), T(nunroll*dt)]
function loss_posteriori_Z(p)
    i0 = @Zygote.ignore rand(1:(length(saveat)-nunroll))
    prob = ODEProblem(dudt_nn_Z, sol_les.u[i0], tspan, p)
    pred = Array(solve(prob, RK4(); u0 = sol_les.u[i0], p = p, saveat = saveat_loss))
    # remember to discard sol at i0
    return T(sum(abs2, stack(sol_les.u[i0+1:i0+nunroll]) - pred))
end

callback(θ_Z, loss_posteriori_Z(θ_Z))

optf = Optimization.OptimizationFunction((x,p)->loss_posteriori_Z(x), Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ_Z)

result_posteriori_Z, time_posteriori_Z, alloc_posteriori_Z, gc_posteriori_Z, mem_posteriori_Z = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1),
    callback = callback,
    maxiters = 50,
)
θ_posteriori_Z = result_posteriori_Z.u


# show (1) loss and (2) time comparing:
#   - priori-Zygote 
#   - priori-Enzyme
#   - posteriori-Zygote
#   - posteriori-Enzyme




# from here ....
# 



# Define a posteriori loss function that calls the ODE solver
# First, make a shorter run
# and remember to set a small dt
dt = T(1e-3);
trange = [T(0), T(2e-3)]
saveat = [dt, 2dt];
u0stacked = stack(ustart);
prob = ODEProblem{true}(dudt_nn, u0stacked, trange; p = P)
ode_data = Array(solve(prob, RK4(); u0 = u0stacked, p = P, saveat = saveat))
ode_data += T(0.1) * rand(Float32, size(ode_data))

# the loss has to be in place 
function loss(
    l::Vector{Float32},
    P,
    u0::Array{Float32},
    tspan::Vector{Float32},
    t::Vector{Float32},
)
    myprob = ODEProblem{true}(dudt_nn, u0, tspan, P)
    pred = Array(solve(myprob, RK4(); u0 = u0, p = P, saveat = t))
    l .= Float32(sum(abs2, ode_data - pred))
    nothing
end
l = [T(0.0)];
loss(l, P, u0stacked, trange, saveat);
l

# Test if the loss can be autodiffed
# [!] dl is called the 'seed' and it has to be marked to be one for correct gradient
l = [T(0.0)];
dl = Enzyme.make_zero(l) .+ T(1);
dP = Enzyme.make_zero(P);
du = Enzyme.make_zero(u0stacked);
@timed Enzyme.autodiff(
    Enzyme.Reverse,
    loss,
    DuplicatedNoNeed(l, dl),
    DuplicatedNoNeed(P, dP),
    DuplicatedNoNeed(u0stacked, du),
    Const(trange),
    Const(saveat),
)
dP.θ

println("Now defining the gradient function")
extra_par = [u0stacked, trange, saveat, du, dP, P];
Textra = typeof(extra_par);
function loss_gradient(G, extra_par)
    u0, trange, saveat, du0, dP, P = extra_par
    # [!] Notice that we are updating P.θ in-place in the loss function
    # Reset gradient to zero
    Enzyme.make_zero!(dP)
    # And remember to pass the seed to the loss function with the dual part set to 1
    Enzyme.autodiff(
        Enzyme.Reverse,
        loss,
        DuplicatedNoNeed([T(0)], [T(1)]),
        DuplicatedNoNeed(P, dP),
        DuplicatedNoNeed(u0, du0),
        Const(trange),
        Const(saveat),
    )
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
    loss(l, p, u0stacked, trange, saveat)
    return l
end
callback = function (θ, l; doplot = false)
    println(l)
    return false
end
callback(P, over_loss(P.θ, P))

using SciMLSensitivity, Optimization, OptimizationOptimisers, Optimisers
optf = Optimization.OptimizationFunction(
    (p, u) -> over_loss(p, u[end]);
    grad = (G, p, e) -> loss_gradient(G, e),
)
optprob = Optimization.OptimizationProblem(optf, P.θ, extra_par)

result_e, time_e, alloc_e, gc_e, mem_e = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 100,
)
