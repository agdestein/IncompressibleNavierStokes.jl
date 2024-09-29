"""
Create dataloader that uses a batch of `batchsize` random samples from
`data` at each evaluation.
The batch is moved to `device`.
"""
create_dataloader_prior(data; batchsize = 50, device = identity) = function dataloader(rng)
    x, y = data
    nsample = size(x)[end]
    d = ndims(x)
    i = sort(shuffle(rng, 1:nsample)[1:batchsize])
    xuse = device(Array(selectdim(x, d, i)))
    yuse = device(Array(selectdim(y, d, i)))
    (xuse, yuse), rng
end

"""
Create trajectory dataloader.
"""
create_dataloader_post(trajectories; nunroll = 10, device = identity) =
    function dataloader(rng)
        (; u, t) = rand(rng, trajectories)
        nt = length(t)
        @assert nt ≥ nunroll
        istart = rand(rng, 1:nt-nunroll)
        it = istart:istart+nunroll
        (; u = device.(u[it]), t = t[it]), rng
    end

"""
Update parameters `θ` to minimize `loss(dataloader(), θ)` using the
optimiser `opt` for `niter` iterations.

Return the a new named tuple `(; opt, θ, callbackstate)` with
updated state and parameters.
"""
function train(; dataloader, loss, trainstate, niter = 100, callback, callbackstate)
    for i = 1:niter
        (; optstate, θ, rng) = trainstate
        batch, rng = dataloader(rng)
        g, = gradient(θ -> loss(batch, θ), θ)
        optstate, θ = Optimisers.update!(optstate, θ, g)
        trainstate = (; optstate, θ, rng)
        callbackstate = callback(callbackstate, trainstate)
    end
    (; trainstate, callbackstate)
end

"""
Wrap loss function `loss(batch, θ)`.

The function `loss` should take inputs like `loss(f, x, y, θ)`.
"""
create_loss_prior(loss, f) = ((x, y), θ) -> loss(f, x, y, θ)

"""
Create a-priori error.
"""
create_relerr_prior(f, x, y) = θ -> norm(f(x, θ) - y) / norm(y)

"""
Compute MSE between `f(x, θ)` and `y`.

The MSE is further divided by `normalize(y)`.
"""
mean_squared_error(f, x, y, θ; normalize = y -> sum(abs2, y), λ = sqrt(eltype(x)(1e-8))) =
    sum(abs2, f(x, θ) - y) / normalize(y) + λ * sum(abs2, θ)

"""
Create a-posteriori loss function.
"""
function create_loss_post(;
    setup,
    method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
    psolver,
    closure,
    nupdate = 1,
)
    closure_model = wrappedclosure(closure, setup)
    setup = (; setup..., closure_model)
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
Create a-posteriori relative error.
"""
function create_relerr_post(;
    data,
    setup,
    method = RKMethods.RK44(; T = eltype(setup.grid.x[1])),
    psolver,
    closure_model,
    nupdate = 1,
)
    setup = (; setup..., closure_model)
    (; dimension, Iu) = setup.grid
    D = dimension()
    (; u, t) = data
    v = copy.(u[1])
    cache = IncompressibleNavierStokes.ode_method_cache(method, setup)
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

"""
Create a-posteriori symmetry error.
"""
function create_relerr_symmetry_post(;
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
    cache = IncompressibleNavierStokes.ode_method_cache(method, setup)
    function err(θ)
        stepper = IncompressibleNavierStokes.create_stepper(
            method;
            setup,
            psolver,
            u = copy.(u),
            temp = nothing,
            t = T(0),
        )
        stepper_rot = IncompressibleNavierStokes.create_stepper(
            method;
            setup,
            psolver,
            u = rot2stag(copy.(u), g),
            temp = nothing,
            t = T(0),
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
Create a-priori equivariance error.
"""
function create_relerr_symmetry_prior(; u, setup, g = 1)
    (; grid, closure_model) = setup
    (; dimension, Iu) = grid
    D = dimension()
    T = eltype(u[1][1])
    function err(θ)
        e = sum(u) do u
            cr = closure_model(rot2stag(u, g), θ)
            rc = rot2stag(closure_model(u, θ), g)
            a, b = T(0), T(0)
            for α = 1:D
                a += sum(abs2, view(rc[α] - cr[α], Iu[α]))
                b += sum(abs2, view(cr[α], Iu[α]))
            end
            sqrt(a) / sqrt(b)
        end
        e / length(u)
    end
end

"""
Create convergence plot for relative error between `f(x, θ)` and `y`.
At each callback, plot is updated and current error is printed.

If `state` is nonempty, it also plots previous convergence.

If not using interactive GLMakie window, set `displayupdates` to
`true`.
"""
function create_callback(
    err;
    θ,
    callbackstate = (;
        n = 0,
        θmin = θ,
        emin = eltype(θ)(Inf),
        hist = Point2f[],
        ctime = time(),
    ),
    displayref = true,
    displayfig = true,
    displayupdates = false,
    figname = nothing,
    nupdate,
)
    obs = Observable([Point2f(0, 0)])
    fig = lines(obs; axis = (; title = "Relative prediction error", xlabel = "Iteration"))
    displayref && hlines!([1.0f0]; linestyle = :dash)
    obs[] = callbackstate.hist
    displayfig && display(fig)
    function callback(callbackstate, trainstate)
        @reset callbackstate.n += 1
        (; n, hist) = callbackstate
        if n % nupdate == 0
            (; θ) = trainstate
            e = err(θ)
            newtime = time()
            itertime = (newtime - callbackstate.ctime) / nupdate
            @reset callbackstate.ctime = newtime
            @info "Iteration $n \t relative error: $e \t sec/iter: $itertime"
            hist = push!(copy(hist), Point2f(n, e))
            @reset callbackstate.hist = hist
            obs[] = hist
            # n < 30 || autolimits!(fig.axis)
            autolimits!(fig.axis)
            displayupdates && display(fig)
            isnothing(figname) || save(figname, fig)
            if e < callbackstate.emin
                @reset callbackstate.θmin = θ
                @reset callbackstate.emin = e
            end
        end
        callbackstate
    end
    (; callbackstate, callback)
end
