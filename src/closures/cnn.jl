"""
    cnn(;
        setup,
        radii,
        channels,
        activations,
        use_bias,
        channel_augmenter = identity,
        rng = Random.default_rng(),
    )

Create CNN closure model. Return a tuple `(closure, θ)` where `θ` are the initial
parameters and `closure(u, θ)` predicts the commutator error.
"""
function cnn(;
    setup,
    radii,
    channels,
    activations,
    use_bias,
    channel_augmenter = identity,
    rng = Random.default_rng(),
)
    r, c, σ, b = radii, channels, activations, use_bias
    (; grid) = setup
    (; dimension, x) = grid
    D = dimension()

    # Weight initializer
    T = eltype(x[1])
    glorot_uniform_T(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Make sure there are two force fields in output
    @assert c[end] == D

    # Add input channel size
    c = [D; c]

    # Create convolutional closure model
    NN = Chain(
        function (u)
            sz..., _, _ = size(u)
            # for α = 1:D
            #     v = selectdim(u, D + 1, α)
            #     v = (v + circshift(v, ntuple(β -> α == β ? -1 : 0, D + 1))) / 2
            # end
            if D == 2
                a = selectdim(u, 3, 1)
                b = selectdim(u, 3, 2)
                a = (a + circshift(a, (-1, 0, 0))) / 2
                b = (b + circshift(b, (0, -1, 0))) / 2
                a = reshape(a, sz..., 1, :)
                b = reshape(b, sz..., 1, :)
                cat(a, b; dims = 3)
            elseif D == 3
                a = selectdim(u, 4, 1)
                b = selectdim(u, 4, 2)
                c = selectdim(u, 4, 3)
                a = (a + circshift(a, (-1, 0, 0, 0))) / 2
                b = (b + circshift(b, (0, -1, 0, 0))) / 2
                c = (c + circshift(c, (0, 0, -1, 0))) / 2
                a = reshape(a, sz..., 1, :)
                b = reshape(b, sz..., 1, :)
                c = reshape(c, sz..., 1, :)
                cat(a, b, c; dims = 4)
            end
        end,

        # Add padding so that output has same shape as commutator error
        u -> pad_circular(u, sum(r)),

        # Some convolutional layers
        (
            Conv(
                ntuple(α -> 2r[i] + 1, D),
                c[i] => c[i+1],
                σ[i];
                use_bias = b[i],
                init_weight = glorot_uniform_T,
            ) for i ∈ eachindex(r)
        )...,

        # Differentiate output to velocity points
        function (u)
            sz..., _, _ = size(u)
            # for α = 1:D
            #     v = selectdim(u, D + 1, α)
            #     v = (v + circshift(v, ntuple(β -> α == β ? -1 : 0, D + 1))) / 2
            # end
            if D == 2
                a = selectdim(u, 3, 1)
                b = selectdim(u, 3, 2)
                # a = (a + circshift(a, (1, 0, 0, 0))) / 2
                # b = (b + circshift(b, (0, 1, 0, 0))) / 2
                a = circshift(a, (1, 0, 0)) - a
                b = circshift(b, (0, 1, 0)) - b
                a = reshape(a, sz..., 1, :)
                b = reshape(b, sz..., 1, :)
                cat(a, b; dims = 3)
            elseif D == 3
                a = selectdim(u, 4, 1)
                b = selectdim(u, 4, 2)
                c = selectdim(u, 4, 3)
                # a = (a + circshift(a, (1, 0, 0, 0))) / 2
                # b = (b + circshift(b, (0, 1, 0, 0))) / 2
                # c = (c + circshift(c, (0, 0, 1, 0))) / 2
                a = circshift(a, (1, 0, 0, 0)) - a
                b = circshift(b, (0, 1, 0, 0)) - b
                c = circshift(c, (0, 0, 1, 0)) - c
                a = reshape(a, sz..., 1, :)
                b = reshape(b, sz..., 1, :)
                c = reshape(c, sz..., 1, :)
                cat(a, b, c; dims = 4)
            end
        end,
    )

    # Create parameter vector (empty state)
    params, state = Lux.setup(rng, NN)
    θ = ComponentArray(params)

    # Compute closure term for given parameters
    closure(u, θ) = first(NN(u, θ, state))

    closure, θ
end
