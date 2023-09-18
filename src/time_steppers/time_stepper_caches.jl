
"""
    ode_method_cache(method, setup)

Get time stepper cache for the given ODE method.
"""
function ode_method_cache end

function ode_method_cache(::AdamsBashforthCrankNicolsonMethod, setup, V, p)
    c₀ = similar(V)
    c₋₁ = similar(V)
    F = similar(V)
    f = similar(p)
    Δp = similar(p)
    Rr = similar(V)
    b = similar(V)
    b₀ = similar(V)
    b₁ = similar(V)
    yDiff₀ = similar(V)
    yDiff₁ = similar(V)
    Gp₀ = similar(V)

    (; c₀, c₋₁, F, f, Δp, Rr, b, b₀, b₁, yDiff₀, yDiff₁, Gp₀)
end

function ode_method_cache(::OneLegMethod{T}, setup, V, p) where {T}
    (; NV, Np) = setup.grid
    u₋₁ = similar(V)
    p₋₁ = similar(p)
    F = similar(V)
    f = similar(p)
    Δp = similar(p)
    GΔp = similar(V)
    (; u₋₁, p₋₁, F, f, Δp, GΔp)
end

function ode_method_cache(method::ExplicitRungeKuttaMethod{T}, setup, u, p) where {T}
    u₀ = similar.(u)
    ns = nstage(method)
    ku = [similar.(u) for i = 1:ns]
    v = similar.(u)
    F = similar.(u)
    G = similar.(u)
    M = similar(p)
    (; u₀, ku, v, F, M, G)
end

function ode_method_cache(method::ImplicitRungeKuttaMethod{T}, setup, V, p) where {T}
    (; NV, Np, Ω) = setup.grid
    (; G, M) = setup.operators
    (; A, b, c) = method

    Vₙ = similar(V)
    pₙ = similar(p)

    # Number of stages
    s = length(b)

    # Extend the Butcher tableau
    Is = sparse(I, s, s)
    Ω_sNV = kron(Is, spdiagm(Ω))
    A_ext = kron(A, sparse(I, NV, NV))
    b_ext = kron(b', sparse(I, NV, NV))
    c_ext = spdiagm(c)

    Vtotₙ = zeros(T, s * NV)
    ptotₙ = zeros(T, s * Np)
    Qⱼ = zeros(T, s * (NV + Np))

    Fⱼ = zeros(T, s * NV)
    ∇Fⱼ = spzeros(T, s * NV, s * NV)

    fⱼ = zeros(T, s * (NV + Np))

    F = similar(V)
    ∇F = spzeros(T, NV, NV)
    f = similar(p)
    Δp = similar(p)
    Gp = similar(V)

    # Gradient operator (could also use 1 instead of c and later scale the pressure)
    Gtot = kron(A, G)

    # Divergence operator
    Mtot = kron(Is, M)
    yMtot = zeros(T, Np * s)

    # Finite volumes
    Ωtot = kron(ones(T, s), Ω)

    # Iteration matrix
    dfmom = spzeros(T, s * NV, s * NV)
    Z2 = spzeros(T, s * Np, s * Np)
    Z = [dfmom Gtot; Mtot Z2]

    (;
        Vₙ,
        pₙ,
        Vtotₙ,
        ptotₙ,
        Qⱼ,
        Fⱼ,
        ∇Fⱼ,
        fⱼ,
        F,
        ∇F,
        f,
        Δp,
        Gp,
        Is,
        Ω_sNV,
        A_ext,
        b_ext,
        c_ext,
        Gtot,
        Mtot,
        yMtot,
        Ωtot,
        dfmom,
        Z,
    )
end
