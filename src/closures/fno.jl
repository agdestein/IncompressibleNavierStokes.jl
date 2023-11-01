"""
    fno(setup, kmax, c, σ, ψ; rng = Random.default_rng(), kwargs...)

Create FNO closure model. Return a tuple `(closure, θ)` where `θ` are the
initial parameters and `closure(V, θ)` predicts the commutator error.
"""
function fno(setup, kmax, c, σ, ψ; rng = Random.default_rng(), kwargs...)
    (; grid) = setup
    (; dimension, N) = grid

    D = dimension()

    @assert all(==(first(N)), N)

    # Fourier layers
    @assert length(kmax) == length(c) == length(σ)

    # Make sure there are two velocity fields in input and output
    c = [2; c]

    # Create FNO closure model
    NN = Chain(
        # Some Fourier layers
        (
            FourierLayer(dimension, kmax[i], c[i] => c[i+1]; σ = σ[i]) for i ∈ eachindex(σ)
        )...,

        # Put channels in first dimension
        u -> permutedims(u, (D + 1, (1:D)..., D + 2)),

        # Compress with a final dense layer
        Dense(c[end] => 2 * c[end], ψ),
        Dense(2 * c[end] => 2; use_bias = false),

        # Put channels back after spatial dimensions
        u -> permutedims(u, ((2:D+1)..., 1, D + 2)),
    )

    # Create parameter vector (empty state)
    params, state = Lux.setup(rng, NN)
    θ = ComponentArray(params)

    # Compute closure term for given parameters
    closure(u, θ) = first(NN(u, θ, state))

    closure, θ
end

"""
    FourierLayer(dimension, kmax, cin => cout; σ = identity, init_weight = glorot_uniform)

Fourier layer operating on uniformly discretized functions.

Some important sizes:

- `dimension`: Spatial dimension, e.g. `Dimension(2)` or `Dimension(3)`.
- `(nx..., cin, nsample)`: Input size
- `(nx..., cout, nsample)`: Output size
- `nx = fill(n, dimension())`: Number of points in each spatial dimension
- `n ≥ kmax`: Same number of points in each spatial dimension, must be
  larger than cut-off wavenumber
- `kmax`: Cut-off wavenumber
- `nsample`: Number of input samples (treated independently)
"""
struct FourierLayer{N,A,F} <: Lux.AbstractExplicitLayer
    dimension::Dimension{N}
    kmax::Int
    cin::Int
    cout::Int
    σ::A
    init_weight::F
end

FourierLayer(
    dimension,
    kmax,
    ch::Pair{Int,Int};
    σ = identity,
    init_weight = glorot_uniform,
) = FourierLayer(dimension, kmax, first(ch), last(ch), σ, init_weight)

Lux.initialparameters(
    rng::AbstractRNG,
    (; dimension, kmax, cin, cout, init_weight)::FourierLayer,
) = (;
    spatial_weight = init_weight(rng, cout, cin),
    spectral_weights = init_weight(rng, fill(kmax + 1, dimension())..., cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; dimension, kmax, cin, cout)::FourierLayer) =
    cout * cin + (kmax + 1)^dimension() * 2 * cout * cin
Lux.statelength(::FourierLayer) = 0

## Pretty printing
function Base.show(io::IO, (; dimension, kmax, cin, cout, σ)::FourierLayer)
    print(io, "FourierLayer{", dimension(), "}(")
    print(io, kmax)
    print(io, ", ", cin, " => ", cout)
    print(io, "; σ = ", σ)
    print(io, ")")
end

# Pass inputs through Fourier layer
function ((; dimension, kmax, cout, cin, σ)::FourierLayer)(x, params, state)
    # TODO: Check if this is more efficient for
    # size(x) = (cin, nx..., nsample) or
    # size(x) = (nx..., cin, nsample)

    # TODO: Set FFT normalization so that layer is truly grid independent

    # Spatial dimension
    N = dimension()

    nx..., _cin, nsample = size(x)
    @assert _cin == cin "Number of input channels must be compatible with weights"
    @assert all(==(first(nx)), nx) "Fourier layer requires same number of grid points in each dimension"
    @assert kmax ≤ first(nx) "Fourier layer input must be discretized on at least `kmax` points"

    # Destructure params
    # The real and imaginary parts of R are stored in two separate channels
    W = params.spatial_weight
    W = reshape(W, ntuple(Returns(1), N)..., cout, cin)
    R = params.spectral_weights
    R = selectdim(R, N + 3, 1) .+ im .* selectdim(R, N + 3, 2)

    # Spatial part (applied point-wise)
    # Do matrix multiplication manually for now
    # TODO: Make W*x more efficient with Tullio.jl
    y = reshape(x, nx..., 1, cin, :)
    y = sum(W .* y; dims = N + 2)
    y = reshape(y, nx..., cout, :)

    # Spectral part (applied mode-wise)
    #
    # Steps:
    #
    # - go to complex-valued spectral space
    # - chop off high wavenumbers
    # - multiply with weights mode-wise
    # - pad with zeros to restore original shape
    # - go back to real valued spatial representation
    #
    # We do matrix multiplications manually for now
    # TODO: Make R*xhat more efficient with Tullio
    ikeep = ntuple(Returns(1:kmax+1), N)
    nkeep = ntuple(Returns(kmax + 1), N)
    dims = ntuple(identity, N)
    z = fft(x, dims)
    z = z[ikeep..., :, :]
    z = reshape(z, nkeep..., 1, cin, :)
    z = sum(R .* z; dims = N + 2)
    z = reshape(z, nkeep..., cout, :)
    z = pad_zeros(z, ntuple(i -> isodd(i) ? 0 : first(nx) - kmax - 1, 2N); dims)
    z = real.(ifft(z, dims))

    # Outer layer: Activation over combined spatial and spectral parts
    # Note: Even though high wavenumbers are chopped off in `z` and may
    # possibly not be present in the input at all, `σ` creates new high
    # wavenumbers. High wavenumber functions may thus be represented using a
    # sequence of Fourier layers. In this case, the `y`s are the only place
    # where information contained in high input wavenumbers survive in a
    # Fourier layer.
    v = σ.(y .+ z)

    # Fourier layer does not modify state
    v, state
end
