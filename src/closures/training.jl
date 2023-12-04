"""
    createdataloader(data; nuse = 50, device = identity)

Create dataloader that uses a batch of `batchsize` random samples from
`data` at each evaluation.
The batch is moved to `device`.
"""
createdataloader(data; batchsize = 50, device = identity) = function dataloader()
    x, y = data
    nsample = size(x)[end]
    d = ndims(x)
    i = sort(shuffle(1:nsample)[1:batchsize])
    xuse = device(Array(selectdim(x, d, i)))
    yuse = device(Array(selectdim(y, d, i)))
    xuse, yuse
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
    createloss(loss, f)

Wrap loss function `loss(batch, θ)`.

The function `loss` should take inputs like `loss(f, x, y, θ)`.
"""
createloss(loss, f) = ((x, y), θ) -> loss(f, x, y, θ)

"""
    mean_squared_error(f, x, y, θ; normalize = y -> sum(abs2, y), λ = sqrt(eps(eltype(x))))

Compute MSE between `f(x, θ)` and `y`.

The MSE is further divided by `normalize(y)`.
"""
mean_squared_error(f, x, y, θ; normalize = y -> sum(abs2, y), λ = sqrt(eps(eltype(x)))) =
    sum(abs2, f(x, θ) - y) / normalize(y) + λ * sum(abs2, θ) / length(θ)

"""
    relative_error(x, y)

Compute average column relative error between matrices `x` and `y`.
"""
relative_error(x, y) =
    sum(norm(x - y) / norm(y) for (x, y) ∈ zip(eachcol(x), eachcol(y))) / size(x, 2)

"""
    relerr_trajectory(uref, setup)

Processor to compute relative error between `uref` and `u` at each iteration.
"""
relerr_trajectory(uref, setup; nupdate = 1) =
    processor() do state
        (; dimension, x, Ip) = setup.grid
        D = dimension()
        T = eltype(x[1])
        e = Ref(T(0))
        on(state) do (; u, n)
            n % nupdate == 0 || return
            neff = n ÷ nupdate
            a, b = T(0), T(0)
            for α = 1:D
                # @show size(uref[n + 1])
                a += sum(abs2, u[α][Ip] - uref[neff+1][α][Ip])
                b += sum(abs2, uref[neff+1][α][Ip])
            end
            e[] += sqrt(a) / sqrt(b) / (length(uref) - 1)
        end
        e
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
function create_callback(f, x, y; state = Point2f[], display_each_iteration = false)
    istart = isempty(state) ? 0 : Int(first(state[end]))
    obs = Observable([Point2f(0, 0)])
    fig = lines(obs; axis = (; title = "Relative prediction error", xlabel = "Iteration"))
    hlines!([1.0f0]; linestyle = :dash)
    obs[] = state
    display(fig)
    function callback(state, i, θ)
        e = norm(f(x, θ) - y) / norm(y)
        @info "Iteration $i \trelative error: $e"
        state = push!(copy(state), Point2f(istart + i, e))
        obs[] = state
        autolimits!(fig.axis)
        display_each_iteration && display(fig)
        state
    end
end
