"""
    cnn(setup, r, c, σ, b; kwargs...)

Create CNN closure model. Return a tuple `(closure, θ)` where `θ` are the initial
parameters and `closure(u, θ)` predicts the commutator error.
"""
function cnn(setup, r, c, σ, b; channel_augmenter = identity, rng = Random.default_rng())
    (; grid) = setup
    (; dimension, x) = grid
    D = dimension()

    # Weight initializer
    T = eltype(x[1])
    glorot_uniform_T(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Make sure there are two force fields in output
    @assert c[end] == D

    # Add input channel size
    c = [2; c]

    # Create convolutional closure model
    NN = Chain(
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
    )

    # Create parameter vector (empty state)
    params, state = Lux.setup(rng, NN)
    θ = ComponentArray(params)

    # Compute closure term for given parameters
    closure(u, θ) = first(NN(u, θ, state))

    closure, θ
end
