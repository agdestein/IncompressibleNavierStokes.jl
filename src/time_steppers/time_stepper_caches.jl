"""
    $FUNCTIONNAME(method, setup, u, temp)

Get time stepper cache for the given ODE method.
"""
function ode_method_cache end

function ode_method_cache(::AdamsBashforthCrankNicolsonMethod, setup, V, p)
    c₀ = zero(V)
    c₋₁ = zero(V)
    F = zero(V)
    f = zero(p)
    Δp = zero(p)
    Rr = zero(V)
    b = zero(V)
    b₀ = zero(V)
    b₁ = zero(V)
    yDiff₀ = zero(V)
    yDiff₁ = zero(V)
    Gp₀ = zero(V)

    (; c₀, c₋₁, F, f, Δp, Rr, b, b₀, b₁, yDiff₀, yDiff₁, Gp₀)
end

function ode_method_cache(::OneLegMethod{T}, setup, u) where {T}
    unew = zero.(u)
    pnew = zero(u[1])
    div = zero(u[1])
    F = zero.(u)
    Δp = zero(u[1])
    (; unew, pnew, F, div, Δp)
end

function ode_method_cache(method::ExplicitRungeKuttaMethod, setup, u, temp)
    u₀ = zero.(u)
    ns = length(method.b)
    ku = [zero.(u) for i = 1:ns]
    div = zero(u[1])
    p = zero(u[1])
    if isnothing(temp)
        temp₀ = nothing
        ktemp = nothing
        diff = nothing
    else
        temp₀ = copy(temp)
        ktemp = [copy(temp) for i = 1:ns]
        diff = zero.(u)
    end
    (; u₀, ku, div, p, temp₀, ktemp, diff)
end

function ode_method_cache(method::ImplicitRungeKuttaMethod{T}, setup, V, p) where {T}
    (; NV, Np, Ω) = setup.grid
    (; G, M) = setup.operators
    (; A, b, c) = method

    Vₙ = zero(V)
    pₙ = zero(p)

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

    F = zero(V)
    ∇F = spzeros(T, NV, NV)
    f = zero(p)
    Δp = zero(p)
    Gp = zero(V)

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
