function tensorclosure(model, u, θ, setup)
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
