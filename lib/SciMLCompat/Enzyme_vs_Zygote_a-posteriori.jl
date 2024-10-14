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
using SciMLSensitivity
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

# get the filtered solution
u_filtered = []
for i in 1:length(sol_dns.u)
    u = (sol_dns.u[i][:,:,1], sol_dns.u[i][:,:,2])
    push!(u_filtered, stack(apply_filter(Φ, u)))
end



#####################
# A posteriori training
#####################
# The filtered dns force can be used to evaulate the overall performance of the NN
all_F_filtered = []
for i in 1:length(saveat)
    F_dns = F_out_dns(sol_dns.u[i], nothing, 0)
    F_filt = apply_filter(Φ, (F_dns[:,:,1],F_dns[:,:,2]))
    push!(all_F_filtered, stack(F_filt))
end

# define a dummy_NN to train
dummy_NN = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
)
θ_0, st = Lux.setup(rng, dummy_NN)
θ_0 = ComponentArray(θ_0)
# and a variation that adds a view at the end
dummy_NN_v = Lux.Chain(
    x -> view(x, :, :, :, :),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    Lux.Conv((3, 3), 2 => 2; pad = (1, 1), stride = (1, 1)),
    x -> view(x, :),
)
θ_v, st_v = Lux.setup(rng, dummy_NN_v)
θ_v = ComponentArray(θ_v)
# set same initial conditions for the dummy NN
θ_v.layer_2 = θ_0.layer_2
θ_v.layer_3 = θ_0.layer_3
θ_v.layer_4 = θ_0.layer_4

# Define the right hand side function with the neural network closure   
dudt_nn_Z(u, θ, t) = begin
    F_out_les(u, nothing, t) .+ Lux.apply(dummy_NN, u, θ, st)[1][:,:,:,1]
end

# Define a callback function to print the loss
callback = function (θ, l; doplot = false)
    println("Loss: ", l)
    return false
end

# Define a total loss function 
function total_loss_Z(p)
    l = 0
    d = 0
    for i in 1:length(saveat)
        l += sum(abs2, all_F_filtered[i] .- dudt_nn_Z(sol_les.u[i], p, 0.0))
        d += sum(abs2, all_F_filtered[i])
    end
    return l/d
end

# Now for Enzyme
# [!] Here we want to compare different methods to define the right hand side function with the closure. They perform differently when used in combination with Enzyme.
dudt_nn_E1(du, u, θ, t) = begin
    F_in_les(du, u, nothing, t)
    view(du, :) .= view(du, :) .+ view(Lux.apply(dummy_NN, u, θ, st)[1][:,:,:,1], :)
    nothing
end
dudt_nn_E2(du, u, θ, t) = begin
    F_in_les(du, u, nothing, t)
    tmp = Lux.apply(dummy_NN, u, θ, st)[1][:,:,:,1]
    @. du += tmp
    nothing
end
function _create_dudt_nn_E(u0) 
    tmp = similar(u0)
    function f(du, u , θ, t)
        F_in_les(du, u, nothing, t)
        tmp .= Lux.apply(dummy_NN, u, θ, st)[1][:,:,:,1]
        @. du += tmp
    end
end
dudt_nn_E3 = _create_dudt_nn_E(stack(u0_les))
dudt_nn_E4(du, u, θ, t) = begin
    F_in_les(du, u, nothing, t)
    view(du, :) .= view(du, :) .+ Lux.apply(dummy_NN_v, u, θ, st_v)[1]
    nothing
end


# Sanity check #1
# compare the time and memory of the three methods
tmp = similar(stack(u0_les))
_, t1, m1, _, _ = @timed for i in 1:100 
    dudt_nn_E1(tmp, u_filtered[i], θ_0, 0.0)
end
tmp1 = copy(tmp)
_, t2, m2, _, _ = @timed for i in 1:100 
    dudt_nn_E2(tmp, u_filtered[i], θ_0, 0.0)
end
tmp2 = copy(tmp)
_, t3, m3, _, _ = @timed for i in 1:100
    dudt_nn_E3(tmp, u_filtered[i], θ_0, 0.0)
end
tmp3 = copy(tmp)
_, t4, m4, _, _ = @timed for i in 1:100
    dudt_nn_E4(tmp, u_filtered[i], θ_v, 0.0)
end
tmp4 = copy(tmp)
@assert tmp1 ≈ tmp2
@assert tmp1 ≈ tmp3
@assert tmp1 ≈ tmp4

# also compare with the out-of-place version
u0 = stack(u0_les);
temp = similar(u0);
dudt_nn_E(temp, u0, θ_0, 0.0);
@assert dudt_nn_Z(u0, θ_0, 0.0) ≈ temp
_, time_out, mem_out, _, _ = @timed for i in 1:100
    dudt_nn_Z(stack(u0_les), θ_0, T(0))
end

# and plot the results
p1 = Plots.bar(["Method 1", "Method 2", "Method 3", "Method 4", "Out-of-place"], [t1, t2, t3, t4, time_out], xlabel = "Method", ylabel = "Time (s)", title = "Single force step calculation")
p2 = Plots.bar(["Method 1", "Method 2", "Method 3", "Method 4", "Out-of-place"], [m1, m2, m3, m4, mem_out], xlabel = "Method", ylabel = "Memory (bytes)", title = "Single force step calculation")
Plots.plot(p1, p2, layout=(2,1), size=(600, 800))


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

# Sanity check #2
# check how the different methods affect SciML
tspan = [T(0), T(0.01)]
prob_in1 = ODEProblem{true}(dudt_nn_E1, u0, tspan, θ_0)
prob_in2 = ODEProblem{true}(dudt_nn_E2, u0, tspan, θ_0)
prob_in3 = ODEProblem{true}(dudt_nn_E3, u0, tspan, θ_0)
prob_in4 = ODEProblem{true}(dudt_nn_E4, u0, tspan, θ_v)
prob_out = ODEProblem(dudt_nn_Z, u0, tspan, θ_0)
x_in1, t_in1, m_in1, _, _ = @timed Array(solve(prob_in1, RK4(); u0 = u0, p = θ_0, saveat = T(0.005), dt=T(0.001)))
x_in2, t_in2, m_in2, _, _ = @timed Array(solve(prob_in2, RK4(); u0 = u0, p = θ_0, saveat = T(0.005), dt=T(0.001)))
x_in3, t_in3, m_in3, _, _ = @timed Array(solve(prob_in3, RK4(); u0 = u0, p = θ_0, saveat = T(0.005), dt=T(0.001)))
x_in4, t_in4, m_in4, _, _ = @timed Array(solve(prob_in4, RK4(); u0 = u0, p = θ_v, saveat = T(0.005), dt=T(0.001)))
x_out, t_out, m_out, _, _ = @timed Array(solve(prob_out, RK4(); u0 = u0, p = θ_0, saveat = T(0.005), dt=T(0.001)))
p1 = Plots.bar(["Method 1", "Method 2", "Method 3", "Method 4", "Out-of-place"], [t_in1, t_in2, t_in3, t_in4, t_out], xlabel = "Method", ylabel = "Time (s)", title = "SciML ODE solution")
p2 = Plots.bar(["Method 1", "Method 2", "Method 3", "Method 4", "Out-of-place"], [m_in1, m_in2, m_in3, m_in4, m_out], xlabel = "Method", ylabel = "Memory (bytes)", title = "SciML ODE solution")
Plots.plot(p1, p2, layout=(2,1), size=(600, 800))

@assert x_in1 ≈ x_in2
@assert x_in1 ≈ x_in3
@assert x_in1 ≈ x_in4
@assert x_in1 ≈ x_out
# [!] Check the assertion above, which you can visualize with the following heatmap: why is there a difference on the boundaries?
Plots.heatmap(x_in[:,:,1,2]-x_out[:,:,1,2])


########################
# define the loss function
nunroll = 5
saveat_loss = [i*dt for i in 1:nunroll]
tspan = [T(0), T(nunroll*dt)]
function loss_posteriori_Z(p)
    i0 = @Zygote.ignore rand(1:(length(saveat)-nunroll))
    prob = ODEProblem(dudt_nn_Z, u_filtered[i0], tspan, p)
    pred = Array(solve(prob, RK4(); u0 = u_filtered[i0], p = p, saveat = saveat_loss))
    # remember to discard sol at i0
    return T(sum(abs2, stack(u_filtered[i0+1:i0+nunroll]) - pred)/ sum(abs2, stack(u_filtered[i0+1:i0+nunroll])))
end

callback(θ_0, loss_posteriori_Z(θ_0))

optf = Optimization.OptimizationFunction((x,p)->loss_posteriori_Z(x), Optimization.AutoZygote()
)
optprob = Optimization.OptimizationProblem(optf, θ_0)

result_posteriori_Z, time_posteriori_Z, alloc_posteriori_Z, gc_posteriori_Z, mem_posteriori_Z = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.1),
    callback = callback,
    maxiters = 50,
)
Zygote.gradient(loss_posteriori_Z, result_posteriori_Z.u)
θ_posteriori_Z = result_posteriori_Z.u


SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true
# ***
# A posteriori with Enzyme
# select the closure
dudt_nn_E = dudt_nn_E4
function wrapper_loss_posteriori_E(p, extra_par)
    l, u, tspan, saveat_loss, i0_ref, u0, tg, _, _ = extra_par
    #l, i0_ref = extra_par
    i0_ref[] = rand(1:(length(saveat)-nunroll))
    @views u0 = u[:,:,:,i0_ref[]]
    # remember to discard sol at i0
    @views tg = u[:,:,:,i0_ref[]+1:i0_ref[]+nunroll] 

    loss_posteriori_E(l, p, u0, tg, tspan, saveat_loss)
    l[1]
end
function loss_posteriori_E(l, p, u0, tg, tspan, saveat_loss)
    prob = ODEProblem{true}(dudt_nn_E, u0, tspan, p)
    pred = Array(solve(prob, RK4(); u0 = u0, p = p, saveat = saveat_loss))
    l .= T(sum(abs2, tg - pred)/ sum(abs2, tg))
    nothing
end

l = [T(0.0)];
dl = Enzyme.make_zero(l) .+ T(1);
dθ = Enzyme.make_zero(θ_0);
i0 = rand(1:(length(saveat)-nunroll))
u0 = u_filtered[i0]
du0 = Enzyme.make_zero(u0);
tg = stack(u_filtered[i0+1:i0+nunroll])
dtg = Enzyme.make_zero(tg);
u = stack(u_filtered)
extra_par = [l, u, tspan, saveat_loss, Ref(i0), u0, tg, du0, dθ];
#extra_par = [l, Ref(i0)];
wrapper_loss_posteriori_E(θ_v, extra_par)

@timed Enzyme.autodiff(
    Enzyme.Reverse,
    loss_posteriori_E,
    DuplicatedNoNeed(l, dl),
    DuplicatedNoNeed(θ_0, dθ),
    DuplicatedNoNeed(u0, du0),
    Const(tg),
    Const(tspan),
    Const(saveat_loss),
)
dθ

function loss_gradient(G, θ, extra_par)
#    l, _, tspan, saveat_loss, i0_ref, u0, tg, du0, dθ = extra_par
#    l, _, tspan, saveat_loss, i0_ref, u0, tg, du0, dθ = extra_par
    l, i0_ref = extra_par
    println("l: ", l)
    println("i0_ref: ", i0_ref[])

    # Reset gradient to zero
    Enzyme.make_zero!(dθ)
    Enzyme.make_zero!(du0)
    # And remember to pass the seed to the loss function with the dual part set to 1
    Enzyme.autodiff(
        Enzyme.Reverse,
        loss_posteriori_E,
        DuplicatedNoNeed([T(0)], [T(1)]),
        DuplicatedNoNeed(θ, dθ),
        DuplicatedNoNeed(u[:,:,:,i0_ref[]], du0),
        Const(u[:,:,:,i0_ref[]+1:i0_ref[]+nunroll]),
        Const(tspan),
        Const(saveat_loss),
    )
    # The gradient matters only for theta
    G .= dθ
    println("G: ", G[1])
    println("θ: ", θ[1])
    nothing
end

# Trigger the gradient
G = copy(dθ);
loss_gradient(G, θ_E, extra_par)

optf = Optimization.OptimizationFunction(
    (u, _) -> wrapper_loss_posteriori_E(u, extra_par),
    grad = (G, u, _) -> loss_gradient(G, u, extra_par),
    Optimization.AutoEnzyme()
)
optprob = Optimization.OptimizationProblem(optf, θ_E, nothing)

result_posteriori_E, time_posteriori_E, alloc_posteriori_E, gc_posteriori_E, mem_posteriori_E = @timed Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 20,
)
θ_posteriori_E = result_posteriori_E.u
@assert result_posteriori_E.u != θ_E


# show (1) loss and (2) time comparing:
#   - priori-Zygote 
#   - priori-Enzyme
#   - posteriori-Zygote
#   - posteriori-Enzyme
