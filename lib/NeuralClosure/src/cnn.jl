"""
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
    T = eltype(x[1])

    # dx = map(d -> d[2:end-1], Δu)

    # Weight initializer
    glorot_uniform_T(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Make sure there are two force fields in output
    @assert c[end] == D

    # Add input channel size
    c = [D; c]

    # Add padding so that output has same shape as commutator error
    padder = ntuple(α -> (u -> pad_circular(u, sum(r); dims = α)), D)

    # Some convolutional layers
    convs = map(
        i -> Conv(
            ntuple(α -> 2r[i] + 1, D),
            c[i] => c[i+1],
            σ[i];
            use_bias = b[i],
            init_weight = glorot_uniform_T,
        ),
        eachindex(r),
    )

    # Create convolutional closure model
    create_closure(collocate, padder, convs..., decollocate; rng)
end
