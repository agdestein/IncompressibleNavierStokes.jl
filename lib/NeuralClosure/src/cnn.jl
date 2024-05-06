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
    (; T, grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()

    # dx = map(d -> d[2:end-1], Δu)

    # Weight initializer
    glorot_uniform_T(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Make sure there are two force fields in output
    @assert c[end] == D

    # Add input channel size
    c = [D; c]

    # Create convolutional closure model
    layers = (
        # Put inputs in pressure points
        collocate,

        # Add padding so that output has same shape as commutator error
        ntuple(
            α ->
                boundary_conditions[α][1] isa PeriodicBC ?
                u -> pad_circular(u, sum(r); dims = α) :
                u -> pad_repeat(u, sum(r); dims = α),
            D,
        ),

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
        decollocate,
    )

    create_closure(layers...; rng)
end
