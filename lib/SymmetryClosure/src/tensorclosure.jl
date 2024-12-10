function tensorclosure(u, θ, model, setup)
    (; Δ) = setup.grid
    @assert all(Δi -> all(≈(Δi[1]), Δi), Array.(Δ))
    Δx = Array(Δ[1])[1]
    B, V = tensorbasis(u, setup)
    x = model(V, θ)
    x .*= Δx^2
    # τ = sum(x .* B; dims = ndims(B))
    τ = IncompressibleNavierStokes.lastdimcontract(x, B, setup)
    τ = apply_bc_p(τ, zero(eltype(u)), setup)
    IncompressibleNavierStokes.divoftensor(τ, setup)
end

tensorclosure(model, setup) = (u, θ) -> tensorclosure(u, θ, model, setup)

function polynomial(V, θ)
    s..., nV = size(V)
    V = eachslice(V; dims = ndims(V))
    basis = if nV == 2
        cat(V[1], V[2], V[1] .* V[2], V[1] .^ 2, V[2] .^ 2; dims = length(s) + 1)
    elseif nV == 5
        cat(
            V[1],
            V[2],
            V[3],
            V[4],
            V[5],
            V[1] .* V[2],
            V[1] .* V[3],
            V[1] .* V[4],
            V[1] .* V[5],
            V[2] .* V[3],
            V[2] .* V[4],
            V[2] .* V[5],
            V[3] .* V[4],
            V[3] .* V[5],
            V[4] .* V[5],
            V[1] .^ 2,
            V[2] .^ 2,
            V[3] .^ 2,
            V[4] .^ 2,
            V[5] .^ 2;
            dims = length(s) + 1,
        )
    end
    θ = reshape(θ, ntuple(Returns(1), length(s))..., size(θ)...)
    x = sum(θ .* basis; dims = ndims(basis))
    reshape(x, s..., size(x, ndims(x)))
end

function create_cnn(setup, radii, channels, activations, use_bias, rng)
    r, c, σ, b = radii, channels, activations, use_bias
    (; grid) = setup
    (; dimension, x) = grid
    D = dimension()
    T = eltype(x[1])

    # dx = map(d -> d[2:end-1], Δu)

    # Weight initializer
    glorot_uniform_T(rng::AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    # Correct output channels
    if D == 2
        @assert c[end] == 3 # 3 basis tensors
    elseif D == 3
        @assert c[end] == 18 # 18 basis tensors
    end

    cin = if D == 2
        2 # 2 invariants
    elseif D == 3
        5 # 5 invariants
    end

    # Add input channel size
    c = [cin; c]

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
    model = NeuralClosure.create_closure(padder, convs...; rng)

    inside = setup.grid.Ip

    function cnn_coeffs(V, θ)
        s = size(V)
        V = V[inside] # Remove boundaries
        V = reshape(V, size(V)..., 1) # Add sample dim
        coeffs = model(V, θ)
        coeffs = pad_circular(coeffs, 1) # Add boundaries
        coeffs = reshape(coeffs, s) # Remove sample dim
    end
end
