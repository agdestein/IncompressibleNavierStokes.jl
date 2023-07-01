
"""
    ode_method_cache(method, setup)

Get time stepper cache for the given ODE method.
"""
function ode_method_cache end

function ode_method_cache(::AdamsBashforthCrankNicolsonMethod{T}, setup) where {T}
    (; NV, Np) = setup.grid

    cₙ = zeros(T, NV)
    cₙ₋₁ = zeros(T, NV)
    F = zeros(T, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    Rr = zeros(T, NV)
    b = zeros(T, NV)
    bₙ = zeros(T, NV)
    bₙ₊₁ = zeros(T, NV)
    yDiffₙ = zeros(T, NV)
    yDiffₙ₊₁ = zeros(T, NV)
    Gpₙ = zeros(T, NV)

    # Compute factorization at first time step (guaranteed since Δt > 0)
    Δt = Ref(T(0))
    Diff_fact = Ref(cholesky(spzeros(0, 0)))

    (;
        cₙ,
        cₙ₋₁,
        F,
        f,
        Δp,
        Rr,
        b,
        bₙ,
        bₙ₊₁,
        yDiffₙ,
        yDiffₙ₊₁,
        Gpₙ,
        Diff_fact,
        Δt,
    )
end

function ode_method_cache(::OneLegMethod{T}, setup) where {T}
    (; NV, Np) = setup.grid
    Vₙ₋₁ = zeros(T, NV)
    pₙ₋₁ = zeros(T, Np)
    F = zeros(T, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    GΔp = zeros(T, NV)
    (; Vₙ₋₁, pₙ₋₁, F, f, Δp, GΔp)
end

function ode_method_cache(method::ExplicitRungeKuttaMethod{T}, setup) where {T}
    (; NV, Np) = setup.grid

    ns = nstage(method)
    kV = zeros(T, NV, ns)
    kp = zeros(T, Np, ns)
    Vtemp = zeros(T, NV)
    Vtemp2 = zeros(T, NV)
    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)

    # Get coefficients of RK method
    (; A, b, c) = method

    (; kV, kp, Vtemp, Vtemp2, F, ∇F, f, Δp)
end

function ode_method_cache(method::ImplicitRungeKuttaMethod{T}, setup) where {T}
    (; NV, Np, Ω) = setup.grid
    (; G, M) = setup.operators
    (; A, b, c) = method

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

    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    Gp = zeros(T, NV)

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
