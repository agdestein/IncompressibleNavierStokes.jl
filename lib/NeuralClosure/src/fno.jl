"""
Create FNO closure model. Return a tuple `(closure, θ)` where `θ` are the
initial parameters and `closure(V, θ)` predicts the commutator error.
"""
function fno(; setup, kmax, c, σ, ψ, rng = Random.default_rng(), kwargs...)
    (; grid) = setup
    (; dimension, x, N) = grid

    D = dimension()

    @assert all(==(first(N)), N)

    # Fourier layers
    @assert length(kmax) == length(c) == length(σ)

    # Make sure there are two velocity fields in input and output
    c = [D; c]

    # Weight initializer
    T = eltype(x[1])
    init_weight(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Create FNO closure model
    layers = (
        # Put inputs in pressure points
        collocate,

        # Conv(ntuple(Returns(1), D), D => c[1]; use_bias = false, init_weight),

        # Some Fourier layers
        (
            FourierLayer(dimension, kmax[i], c[i] => c[i+1]; σ = σ[i], init_weight) for
            i ∈ eachindex(σ)
        )...,

        # Compress with a final dense layer
        # Conv(ntuple(Returns(1), D), c[end] => D; use_bias = false, init_weight),
        Conv(ntuple(Returns(1), D), c[end] => 2 * c[end], ψ; init_weight),
        Conv(ntuple(Returns(1), D), 2 * c[end] => D; use_bias = false, init_weight),

        # Differentiate output to velocity points
        decollocate,
    )
    create_closure(layers...; rng)
end

"""
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
struct FourierLayer{D,A,F} <: Lux.AbstractLuxLayer
    dimension::Dimension{D}
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
    # spectral_weights = init_weight(rng, fill(kmax + 1, dimension())..., cout, cin, 2),
    spectral_weights = init_weight(rng, fill(2 * (kmax + 1), dimension())..., cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
# Lux.parameterlength((; dimension, kmax, cin, cout)::FourierLayer) =
#     cout * cin + (kmax + 1)^dimension() * 2 * cout * cin
Lux.parameterlength((; dimension, kmax, cin, cout)::FourierLayer) =
    cout * cin + (2 * (kmax + 1))^dimension() * 2 * cout * cin
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
    D = dimension()

    nx..., _cin, nsample = size(x)
    K = nx[1]
    @assert _cin == cin "Number of input channels must be compatible with weights"
    @assert all(==(first(nx)), nx) "Fourier layer requires same number of grid points in each dimension"
    @assert kmax ≤ first(nx) "Fourier layer input must be discretized on at least `kmax` points"

    # Destructure params
    # The real and imaginary parts of R are stored in two separate channels
    W = params.spatial_weight
    R = params.spectral_weights
    R = selectdim(R, D + 3, 1) .+ im .* selectdim(R, D + 3, 2)

    # Spatial part (applied point-wise)
    error("Replace Tullio")
    # if D == 2
    #     @tullio y[i₁, i₂, b, s] := W[b, a] * x[i₁, i₂, a, s]
    # elseif D == 3
    #     @tullio y[i₁, i₂, i₃, b, s] := W[b, a] * x[i₁, i₂, i₃, a, s]
    # end

    # Spectral part (applied mode-wise)
    #
    # Steps:
    #
    # - go to complex-valued spectral space
    # - chop off high wavenumbers
    # - multiply with weights mode-wise
    # - pad with zeros to restore original shape
    # - go back to real valued spatial representation
    ikeep = ntuple(Returns([1:kmax+1; K-kmax:K]), D)
    # ikeep = ntuple(Returns(1:kmax+1), D)
    dims = ntuple(identity, D)
    xhat = fft(x, dims)
    xhat = xhat[ikeep..., :, :]
    if D == 2
        error("Replace Tullio")
        # @tullio z[k₁, k₂, b, s] := R[k₁, k₂, b, a] * xhat[k₁, k₂, a, s]
        z = cat(
            z[1:kmax+1, :, :, :],
            zero(similar(z, K - 2 * (kmax + 1), 2 * (kmax + 1), cout, nsample)),
            z[end-kmax:end, :, :, :];
            dims = 1,
        )
        z = cat(
            z[:, 1:kmax+1, :, :],
            zero(similar(z, K, K - 2 * (kmax + 1), cout, nsample)),
            z[:, end-kmax:end, :, :];
            dims = 2,
        )
    elseif D == 3
        error("Replace Tullio")
        # @tullio z[k₁, k₂, k₃, b, s] := R[k₁, k₂, k₃, b, a] * xhat[k₁, k₂, k₃, a, s]
        z = cat(
            z[1:kmax+1, :, :, :, :],
            zero(
                similar(
                    z,
                    K - 2 * (kmax + 1),
                    2 * (kmax + 1),
                    2 * (kmax + 1),
                    cout,
                    nsample,
                ),
            ),
            z[end-kmax:end, :, :, :, :];
            dims = 1,
        )
        z = cat(
            z[:, 1:kmax+1, :, :, :],
            zero(similar(z, K, K - 2 * (kmax + 1), 2 * (kmax + 1), cout, nsample)),
            z[:, end-kmax:end, :, :, :];
            dims = 2,
        )
        z = cat(
            z[:, :, 1:kmax+1, :, :],
            zero(similar(z, K, K, K - 2 * (kmax + 1), cout, nsample)),
            z[:, :, end-kmax:end, :, :];
            dims = 3,
        )
    end
    # z = pad_zeros(z, ntuple(i -> isodd(i) ? 0 : first(nx) - kmax - 1, 2 * D); dims)
    z = real.(ifft(z, dims))

    # Outer layer: Activation over combined spatial and spectral parts
    # Note: Even though high wavenumbers are chopped off in `z` and may
    # possibly not be present in the input at all, `σ` creates new high
    # wavenumbers. High wavenumber functions may thus be represented using a
    # sequence of Fourier layers. In this case, the `y`s are the only place
    # where information contained in high input wavenumbers survive in a
    # Fourier layer.
    v = σ.(y .+ z)

    # @infiltrate

    # Fourier layer does not modify state
    v, state
end
