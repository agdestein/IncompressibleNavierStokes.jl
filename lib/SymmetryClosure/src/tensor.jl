"Compute symmetry tensor basis (differentiable version)."
function tensorbasis(u, setup)
    T = eltype(u)
    D = setup.grid.dimension()
    tensorbasis!(
        ntuple(α -> similar(u, SMatrix{D,D,T,D * D}, setup.grid.N), D == 2 ? 3 : 11),
        ntuple(α -> similar(u, setup.grid.N), D == 2 ? 2 : 5),
        u,
        setup,
    )
end

ChainRulesCore.rrule(::typeof(tensorbasis), u, setup) =
    (tensorbasis(u, setup), φ -> error("Not yet implemented"))

"""
Compute symmetry tensor basis `B[1]`-`B[11]` and invariants `V[1]`-`V[5]`,
as specified in [Silvis2017](@cite) in equations (9) and (11).
Note that `B[1]` corresponds to ``T_0`` in the paper, and `V` to ``I``.
"""
function tensorbasis!(B, V, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu, dimension) = grid
    D = dimension()
    @kernel function basis2!(B, V, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ∇u = ∇(u, I, Δ, Δu)
        S = (∇u + ∇u') / 2
        R = (∇u - ∇u') / 2
        B[1][I] = idtensor(u, I)
        B[2][I] = S
        B[3][I] = S * R - R * S
        V[1][I] = tr(S * S)
        V[2][I] = tr(R * R)
    end
    @kernel function basis3!(B, V, u, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        ∇u = ∇(u, I, Δ, Δu)
        S = (∇u + ∇u') / 2
        R = (∇u - ∇u') / 2
        B[1][I] = idtensor(u, I)
        B[2][I] = S
        B[3][I] = S * R - R * S
        B[4][I] = S * S
        B[5][I] = R * R
        B[6][I] = S * S * R - R * S * S
        B[7][I] = S * R * R + R * R * S
        B[8][I] = R * S * R * R - R * R * S * R
        B[9][I] = S * R * S * S - S * S * R * S
        B[10][I] = S * S * R * R + R * R * S * S
        B[11][I] = R * S * S * R * R - R * R * S * S * R
        V[1][I] = tr(S * S)
        V[2][I] = tr(R * R)
        V[3][I] = tr(S * S * S)
        V[4][I] = tr(S * R * R)
        V[5][I] = tr(S * S * R * R)
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    basis! = D == 2 ? basis2! : basis3!
    basis!(backend, workgroupsize)(B, V, u, I0; ndrange = Np)
    B, V
end
