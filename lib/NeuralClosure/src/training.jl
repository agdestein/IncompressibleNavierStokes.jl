"""
    createdataloader(data; nuse = 50, device = identity, rng)

Create dataloader that uses a batch of `batchsize` random samples from
`data` at each evaluation.
The batch is moved to `device`.
"""
create_dataloader_prior(data; batchsize = 50, device = identity, rng) =
    function dataloader()
        x, y = data
        nsample = size(x)[end]
        d = ndims(x)
        i = sort(shuffle(rng, 1:nsample)[1:batchsize])
        xuse = device(Array(selectdim(x, d, i)))
        yuse = device(Array(selectdim(y, d, i)))
        xuse, yuse
    end

"""
    create_dataloader_post(trajectories; nunroll = 10, device = identity, rng)

Create trajectory dataloader.
"""
create_dataloader_post(trajectories; nunroll = 10, device = identity, rng) =
    function dataloader()
        (; u, t) = rand(trajectories)
        nt = length(t)
        @assert nt ≥ nunroll
        istart = rand(rng, 1:nt-nunroll)
        it = istart:istart+nunroll
        (; u = device.(u[it]), t = t[it])
    end

"""
    train(
        dataloaders,
        loss,
        opt,
        θ;
        niter = 100,
        ncallback = 1,
        callback = (i, θ) -> println("Iteration \$i of \$niter"),
    )

Update parameters `θ` to minimize `loss(dataloader(), θ)` using the
optimiser `opt` for `niter` iterations.

Return the a new named tuple `(; opt, θ, callbackstate)` with
updated state and parameters.
"""
function train(
    dataloaders,
    loss,
    opt,
    θ;
    niter = 100,
    ncallback = 1,
    callback = (state, i, θ) -> println("Iteration $i of $niter"),
    callbackstate = nothing,
)
    for i = 1:niter
        g = sum(dataloaders) do d
            b = d()
            first(gradient(θ -> loss(b, θ), θ))
        end
        opt, θ = Optimisers.update(opt, θ, g)
        if i % ncallback == 0
            callbackstate = callback(callbackstate, i, θ)
        end
    end
    (; opt, θ, callbackstate)
end

"""
    createloss_prior(loss, f)

Wrap loss function `loss(batch, θ)`.

The function `loss` should take inputs like `loss(f, x, y, θ)`.
"""
create_loss_prior(loss, f) = ((x, y), θ) -> loss(f, x, y, θ)

"""
    create_relerr_prior(f, x, y)

Create a-priori error.
"""
create_relerr_prior(f, x, y) = θ -> norm(f(x, θ) - y) / norm(y)

"""
    mean_squared_error(f, x, y, θ; normalize = y -> sum(abs2, y), λ = sqrt(eps(eltype(x))))

Compute MSE between `f(x, θ)` and `y`.

The MSE is further divided by `normalize(y)`.
"""
mean_squared_error(f, x, y, θ; normalize = y -> sum(abs2, y), λ = sqrt(eltype(x)(1e-8))) =
    sum(abs2, f(x, θ) - y) / normalize(y) + λ * sum(abs2, θ)

"""
    create_loss_post(;
        setup,
        method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
        psolver,
        closure,
        nupdate = 1,
        projectorder = :last,
    )

Create a-posteriori loss function.
"""
function create_loss_post(;
    setup,
    method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
    psolver,
    closure,
    nupdate = 1,
    projectorder = :last,
)
    closure_model = wrappedclosure(closure, setup)
    setup = (; setup..., closure_model, projectorder)
    (; dimension, Iu) = setup.grid
    D = dimension()
    function loss_post(data, θ)
        T = eltype(θ)
        (; u, t) = data
        v = u[1]
        stepper = IncompressibleNavierStokes.create_stepper(
            method;
            setup,
            psolver,
            u = v,
            temp = nothing,
            t = t[1],
        )
        loss = zero(eltype(v[1]))
        for it = 2:length(t)
            Δt = (t[it] - t[it-1]) / nupdate
            for isub = 1:nupdate
                stepper = IncompressibleNavierStokes.timestep(method, stepper, Δt; θ)
            end
            a, b = T(0), T(0)
            for α = 1:length(u[1])
                a += sum(abs2, (stepper.u[α]-u[it][α])[Iu[α]])
                b += sum(abs2, u[it][α][Iu[α]])
            end
            loss += a / b
        end
        loss / (length(t) - 1)
    end
end

"""
    create_relerr_post(;
        data,
        setup,
        method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
        psolver,
        closure_model,
        nupdate = 1,
    )

Create a-posteriori relative error.
"""
function create_relerr_post(;
    data,
    setup,
    method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
    psolver,
    closure_model,
    nupdate = 1,
    projectorder = :last,
)
    setup = (; setup..., closure_model, projectorder)
    (; dimension, Iu) = setup.grid
    D = dimension()
    (; u, t) = data
    v = copy.(u[1])
    cache = IncompressibleNavierStokes.ode_method_cache(method, setup, v, nothing)
    function relerr_post(θ)
        T = eltype(u[1][1])
        copyto!.(v, u[1])
        stepper = IncompressibleNavierStokes.create_stepper(
            method;
            setup,
            psolver,
            u = v,
            temp = nothing,
            t = t[1],
        )
        e = zero(T)
        for it = 2:length(t)
            Δt = (t[it] - t[it-1]) / nupdate
            for isub = 1:nupdate
                stepper =
                    IncompressibleNavierStokes.timestep!(method, stepper, Δt; θ, cache)
            end
            a, b = T(0), T(0)
            for α = 1:D
                # a += sum(abs2, (stepper.u[α]-u[it][α])[Iu[α]])
                # b += sum(abs2, u[it][α][Iu[α]])
                a += sum(abs2, view(stepper.u[α] - u[it][α], Iu[α]))
                b += sum(abs2, view(u[it][α], Iu[α]))
            end
            e += sqrt(a) / sqrt(b)
        end
        e / (length(t) - 1)
    end
end

function create_relerr_symmetry(;
    u,
    setup,
    method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
    psolver,
    Δt,
    nstep,
    g = 1,
)
    (; dimension, Iu) = setup.grid
    D = dimension()
    T = eltype(u[1])
    cache = IncompressibleNavierStokes.ode_method_cache(method, setup, copy.(u), nothing)
    function err(θ)
        stepper = IncompressibleNavierStokes.create_stepper(
            method;
            setup,
            psolver,
            u = copy.(u),
            temp = nothing,
            t = T(0)
        )
        stepper_rot = IncompressibleNavierStokes.create_stepper(
            method;
            setup,
            psolver,
            u = rot2stag(copy.(u), g),
            temp = nothing,
            t = T(0)
        )
        e = zero(T)
        for it = 1:nstep
            stepper = IncompressibleNavierStokes.timestep!(method, stepper, Δt; θ, cache)
            stepper_rot =
                IncompressibleNavierStokes.timestep!(method, stepper_rot, Δt; θ, cache)
            u_rot = rot2stag(stepper.u, g)
            a, b = T(0), T(0)
            for α = 1:D
                a += sum(abs2, view(stepper_rot.u[α] - u_rot[α], Iu[α]))
                b += sum(abs2, view(u_rot[α], Iu[α]))
            end
            e += sqrt(a) / sqrt(b)
        end
        e / nstep
    end
end

"""
    create_callback(
        f,
        x,
        y;
        state = Point2f[],
        display_each_iteration = false,
    )

Create convergence plot for relative error between `f(x, θ)` and `y`.
At each callback, plot is updated and current error is printed.

If `state` is nonempty, it also plots previous convergence.

If not using interactive GLMakie window, set `display_each_iteration` to
`true`.
"""
function create_callback(
    err;
    θ,
    callbackstate = (; θmin = θ, emin = eltype(θ)(Inf), hist = Point2f[]),
    displayref = true,
    display_each_iteration = false,
)
    istart = isempty(callbackstate.hist) ? 0 : Int(callbackstate.hist[end][1])
    obs = Observable([Point2f(0, 0)])
    fig = lines(obs; axis = (; title = "Relative prediction error", xlabel = "Iteration"))
    displayref && hlines!([1.0f0]; linestyle = :dash)
    obs[] = callbackstate.hist
    display(fig)
    function callback(state, i, θ)
        e = err(θ)
        @info "Iteration $i \trelative error: $e"
        hist = push!(copy(state.hist), Point2f(istart + i, e))
        obs[] = hist
        # i < 30 || autolimits!(fig.axis)
        autolimits!(fig.axis)
        display_each_iteration && display(fig)
        state = (; state..., hist)
        e < state.emin && (state = (; state..., θmin = θ, emin = e))
        state
    end
    (; callbackstate, callback)
end
