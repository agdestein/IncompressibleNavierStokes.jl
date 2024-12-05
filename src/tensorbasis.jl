"Compute symmetry tensor basis (differentiable version)."
function tensorbasis(u, setup)
    (; N, dimension) = setup.grid
    T = eltype(u)
    D = dimension()
    S = SMatrix{D,D,T,D * D}
    B = similar(u, S, N..., D == 2 ? 3 : 11)
    V = similar(u, N..., D == 2 ? 2 : 5)
    tensorbasis!(fill!(B, zero(S)), fill!(V, 0), u, setup)
end

ChainRulesCore.rrule(::typeof(tensorbasis), u, setup) = (
    tensorbasis(u, setup),
    BV -> (NoTangent(), tensorbasis_adjoint!(zero(u), BV[1], BV[2], u, setup), NoTangent()),
)
"""
Compute symmetry tensor basis `B[1]`-`B[11]` and invariants `V[1]`-`V[5]`,
as specified in [Silvis2017](@cite) in equations (9) and (11).
Note that `B[1]` corresponds to ``T_0`` in the paper, and `V` to ``I``.
"""
function tensorbasis!(B, V, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu, dimension) = grid
    I0 = first(Ip)
    I0 -= oneunit(I0)
    tensorbasis_kernel!(backend, workgroupsize)(dimension, B, V, u, Δ, Δu, I0; ndrange = Np)
    B, V
end

function tensorbasis_adjoint!(ubar, Bbar, Vbar, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu, dimension) = grid
    I0 = first(Ip)
    I0 -= oneunit(I0)
    kernel! = tensorbasis_adjoint_kernel!(backend, workgroupsize)
    kernel!(dimension, ubar, Bbar, Vbar, u, Δ, Δu, I0; ndrange = Np)
    ubar
end

@kernel function tensorbasis_kernel!(::Dimension{2}, B, V, u, Δ, Δu, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    ∇u = ∇(u, I, Δ, Δu)
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    B[I, 1] = idtensor(u, I)
    B[I, 2] = S
    B[I, 3] = S * R - R * S
    V[I, 1] = dot(S, S)
    V[I, 2] = dot(R, R)
end

@kernel function tensorbasis_kernel!(::Dimension{3}, B, V, u, Δ, Δu, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    ∇u = ∇(u, I, Δ, Δu)
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    B[I, 1] = idtensor(u, I)
    B[I, 2] = S
    B[I, 3] = S * R - R * S
    B[I, 4] = S * S
    B[I, 5] = R * R
    B[I, 6] = S * S * R - R * S * S
    B[I, 7] = S * R * R + R * R * S
    B[I, 8] = R * S * R * R - R * R * S * R
    B[I, 9] = S * R * S * S - S * S * R * S
    B[I, 10] = S * S * R * R + R * R * S * S
    B[I, 11] = R * S * S * R * R - R * R * S * S * R
    V[I, 1] = dot(S, S)
    V[I, 2] = dot(R, R)
    V[I, 3] = dot(S, S, S)
    V[I, 4] = dot(S, R, R)
    V[I, 5] = dot(S, S, R, R)
end

@kernel function tensorbasis_adjoint_kernel!(::Dimension{2}, ubar, Bbar, Vbar, u, Δ, Δu, I0)
    I = @index(Global, Cartesian)
    I = I + I0
    ∇u = ∇(u, I, Δ, Δu)
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    Sbar = Bbar[I, 2] + Bbar[I, 3] * R' - R' * Bbar[I, 3] + 2 * Vbar[I, 1] * S
    Rbar = S' * Bbar[I, 3] - Bbar[I, 3] * S' + 2 * Vbar[I, 2] * R
    ∇ubar = (Sbar + Sbar') / 2 + (Rbar - Rbar') / 2
    ∇_adjoint!(∇ubar, ubar, I, Δ, Δu)
    ubar
end

@kernel function tensorbasis_adjoint_kernel!(::Dimension{3}, ubar, Bbar, Vbar, u, Δ, Δu, I0)
    # TODO
end

"""
Compute `c[I] = sum_i a[I, i] * b[I, i]`, where `c[:, i]` and `b[:, i]` are
tensor fields (elements are `SMatrix`es), `a[:, i]` is a scalar field,
and the `i` dimension is contracted (multiple channels).
"""
function lastdimcontract(a, b, setup)
    s..., n = size(a)
    c = similar(b, s)
    lastdimcontract!(c, a, b, setup)
end

ChainRulesCore.rrule(::typeof(lastdimcontract), a, b, setup) = (
    lastdimcontract(a, b, setup),
    function (cbar)
        abar = zero(a)
        bbar = zero(b)
        lastdimcontract_adjoint!(abar, bbar, cbar, a, b, setup)
        (NoTangent(), abar, bbar, NoTangent())
    end,
)

function lastdimcontract!(c, a, b, setup)
    (; backend, workgroupsize) = setup
    s..., n = size(a)
    @assert size(a) == size(b)
    @assert size(c) == s
    lastdimcontract_kernel!(backend, workgroupsize)(c, a, b, n; ndrange = s)
    c
end

function lastdimcontract_adjoint!(abar, bbar, cbar, a, b, setup)
    (; backend, workgroupsize) = setup
    s..., n = size(a)
    @assert size(a) == size(b) == size(abar) == size(bbar)
    @assert size(cbar) == s
    kernel! = lastdimcontract_adjoint_kernel!(backend, workgroupsize)
    kernel!(abar, bbar, cbar, a, b, n; ndrange = s)
    abar, bbar
end

@kernel function lastdimcontract_kernel!(c, a, b, n)
    I = @index(Global, Cartesian)
    cI = zero(eltype(c))
    i = 1
    while i <= n
        cI += a[I, i] * b[I, i]
        i += 1
    end
    c[I] = cI
end

@kernel function lastdimcontract_adjoint_kernel!(abar, bbar, cbar, a, b, n)
    I = @index(Global, Cartesian)
    i = 1
    while i <= n
        # cI += a[I, i] * b[I, i]
        abar[I, i] = dot(cbar[I], b[I, i])
        bbar[I, i] = cbar[I] * a[I, i]
        i += 1
    end
end

function monitor(τ)
    @info "Forward monitor" typeof(τ) size(τ)
    τ
end

ChainRulesCore.rrule(::typeof(monitor), τ) = (monitor(τ), function (τbar)
    @info "Pullback monitor" typeof(τ) typeof(τbar) size(τ) size(τbar)
    (NoTangent(), τbar)
end)
