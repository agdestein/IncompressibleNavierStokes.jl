"""
    train(
        loss,
        opt,
        θ;
        niter = 100,
        ncallback = 1,
        callback = (i, θ) -> println("Iteration \$i of \$niter"),
    )

Update parameters `θ` to minimize `loss(θ)` using the optimiser `opt` for
`niter` iterations.

Return the a new named tuple `(; opt, θ, callbackstate)` with updated state and
parameters.
"""
function train(
    loss,
    opt,
    θ;
    niter = 100,
    ncallback = 1,
    callback = (state, i, θ) -> println("Iteration $i of $niter"),
    callbackstate = nothing,
)
    for i = 1:niter
        g = first(gradient(loss, θ))
        opt, θ = Optimisers.update(opt, θ, g)
        if i % ncallback == 0
            callbackstate = callback(callbackstate, i, θ)
        end
    end
    (; opt, θ, callbackstate)
end

"""
    create_randloss(loss, f, x, y; nuse = size(x, 2), device = identity)

Create loss function `randloss(θ)` that uses a batch of `nuse` random samples from
`(x, y)` at each evaluation.

The function `loss` should take inputs like `loss(f, x, y, θ)`.

The batch is moved to `device` before the loss is evaluated.
"""
function create_randloss(loss, f, x, y; nuse = size(x, 2), device = identity)
    x = reshape(x, size(x, 1), :)
    y = reshape(y, size(y, 1), :)
    nsample = size(x, 2)
    d = ndims(x)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        xuse = Zygote.@ignore device(Array(selectdim(x, d, i)))
        yuse = Zygote.@ignore device(Array(selectdim(y, d, i)))
        loss(f, xuse, yuse, θ)
    end
end

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
