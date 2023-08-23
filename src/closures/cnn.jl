"""
    cnn(setup, r, c, σ, b; kwargs...)

Create CNN closure model. Return a tuple `(closure, Θ)` where `Θ` are the initial
parameters and `closure(V, Θ)` predicts the commutator error.
"""
cnn(setup, r, c, σ, b; kwargs...) = cnn(setup.grid.dimension, setup, r, c, σ, b; kwargs...)

function cnn(
    ::Dimension{2},
    setup,
    r,
    c,
    σ,
    b;
    channel_augmenter = identity,
    rng = Random.default_rng(),
)
    (; grid) = setup
    (; Nx, Ny, x) = grid

    # For now
    T = eltype(x)
    @assert T == Float32

    # Make sure there are two velocity fields in input and output
    @assert c[1] == 2
    # @assert c[1] == 4
    @assert c[end] == 2

    # Create convolutional closure model
    NN = Chain(
        # Unflatten and separate u and v velocities
        V -> reshape(V, Nx, Ny, 2, :),

        # # uu, uv, vu, vv
        # V -> reshape(V, Nx, Ny, 2, 1, :) .* reshape(V, Nx, Ny, 1, 2, :),
        # V -> reshape(V, Nx, Ny, 4, :),

        # Add padding so that output has same shape as commutator error
        u -> pad_circular(u, sum(r)),

        # Some convolutional layers
        (
            Conv((2r[i] + 1, 2r[i] + 1), c[i] => c[i+1], σ[i]; use_bias = b[i]) for
            i ∈ eachindex(r)
        )...,

        # Flatten to vector
        u -> reshape(u, :, size(u, 4)),
    )

    # Create parameter vector (empty state)
    params, state = Lux.setup(rng, NN)
    θ = ComponentArray(params)

    """
        closure(V, θ) 

    Compute closure term for given parameters `θ`.
    """
    function closure end
    closure(V, θ) = first(NN(V, θ, state))
    closure(V::AbstractVector, θ) = reshape(closure(reshape(V, :, 1), θ), :)

    closure, θ
end
