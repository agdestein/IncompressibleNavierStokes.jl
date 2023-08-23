"""
    fno(setup, c, σ, kmax; rng = Random.default_rng(), kwargs...)

Create FNO closure model. Return a tuple `(closure, θ)` where `θ` are the initial
parameters and `closure(V, θ)` predicts the commutator error.
"""
function fno(setup, c, σ, kmax; rng = Random.default_rng(), kwargs...)
    (; grid) = setup
    (; dimension) = grid

    if dimension() == 2
        (; Nx, Ny) = grid
        _nx = (Nx, Ny)
    elseif dimension() == 3
        (; Nx, Ny, Nz) = grid
        _nx = (Nx, Ny, Nz)
    end
    @assert all(==(first(_nx)), _nx)

    # Make sure there are two velocity fields in input and output
    @assert c[1] == 2
    @assert c[end] == 2

    # Create convolutional closure model
    NN = Chain(
        # Unflatten and separate u and v velocities
        V -> reshape(V, _nx..., 2, :),

        # # uu, uv, vu, vv
        # V -> reshape(V, Nx, Ny, 2, 1, :) .* reshape(V, Nx, Ny, 1, 2, :),
        # V -> reshape(V, Nx, Ny, 4, :),

        # Some Fourier layers
        (FourierLayer(dimension, c[i] => c[i+1], kmax; σ = σ[i]) for i ∈ eachindex(r))...,

        # Flatten to vector
        u -> reshape(u, 2 * prod(_nx...), :),
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

"""
    FourierLayer(dimension, cin => cout, kmax; σ = identity, init_weight = glorot_uniform)

Fourier layer operating on uniformly discretized functions.

Some important sizes:

- `dimension`: Spatial dimension, eg `Dimension(2)` or `Dimension(3)`.
- `(cin, nx..., nsample)`: input size
- `(cout, nx..., nsample)`: output size
- `nx = fill(n, dimension())`: number of points in each spatial dimension
- `n ≥ kmax`: Same number of points in each spatial dimension, must be
  larger than cut-off frequency
- `kmax`: Cut-off frequency
- `nsample`: Number of input samples (treated independently)
"""
struct FourierLayer{N,A,F} <: Lux.AbstractExplicitLayer
    dimension::Dimension{N}
    cin::Int
    cout::Int
    kmax::Int
    σ::A
    init_weight::F
end

FourierLayer(
    dimension,
    ch::Pair{Int,Int},
    kmax;
    σ = identity,
    init_weight = glorot_uniform,
) = FourierLayer(dimension, first(ch), last(ch), kmax, σ, init_weight)

Lux.initialparameters(
    rng::AbstractRNG,
    (; dimension, kmax, cin, cout, init_weight)::FourierLayer,
) = (;
    spatial_weight = init_weight(rng, cout, cin),
    spectral_weights = init_weight(rng, cout, cin, fill(kmax + 1, dimension())..., 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; cin, cout, kmax)::FourierLayer) =
    cout * cin + cout * cin * (kmax + 1)^dimension() * 2
Lux.statelength(::FourierLayer) = 0

# Pass inputs through Fourier layer
function ((; cout, cin, kmax, σ)::FourierLayer{N})(x, params, state)
    _cin, nx..., nsample = size(x)
    @assert _cin == cin "Number of input channels must be compatible with weights"
    @assert all(==(first(nx)), nx) "Fourier layer requires same number of grid points in each dimension"
    @assert kmax ≤ first(nx) "Fourier layer input must be discretized on at least `kmax` points"

    # Destructure params
    # The real and imaginary parts of R are stored in two separate channels
    W = params.spatial_weight
    R = params.spectral_weights
    R = selectdim(R, 3 + N, 1) .+ im .* selectdim(R, 3 + N, 2)

    # Spatial part (applied point-wise)
    y = reshape(x, cin, :)
    y = W * y
    y = reshape(y, cout, nx..., :)

    # Spectral part (applied mode-wise)
    # - go to complex-valued spectral space
    # - chop off high frequencies
    # - multiply with weights mode-wise
    # - pad with zeros to restore original shape
    # - go back to real valued spatial representation
    ikeep = ntuple(Returns(1:kmax+1), N)
    z = fft(x, 2:1+N)
    z = z[:, ikeep..., :]
    z = reshape(z, 1, cin, ikeep..., :)
    z = sum(R .* z; dims = 2)
    z = reshape(z, cout, ikeep..., :)
    z = pad_zeros(z, ntuple(i -> isodd(i) ? 0 : first(nx) - kmax - 1, 2N); dims = 2:1+N)
    z = real.(ifft(z, 2:1+N))

    # Outer layer: Activation over combined spatial and spectral parts
    # Note: Even though high frequencies are chopped of in `z` and may
    # possibly not be present in the input at all, `σ` creates new high frequencies.
    # High frequency functions may thus be represented using a sequence of
    # Fourier layers. In this case, the `y`s are the only place where
    # information contained in high
    # input frequencies survive in a Fourier layer.
    v = σ.(y .+ z)

    # Fourier layer does not change state
    v, state
end
