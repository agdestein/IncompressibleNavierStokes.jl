"""
    AbstractODEMethod

Abstract ODE method.
"""
abstract type AbstractODEMethod{T} end

"""
    AdamsBashforthCrankNicolsonMethod(
        T = Float64;
        α₁ = T(3 // 2),
        α₂ = T(-1 // 2),
        θ = T(1 // 2),
        p_add_solve = true,
        method_startup,
    )

IMEX AB-CN: Adams-Bashforth for explicit convection (parameters `α₁` and `α₂`)
and Crank-Nicolson for implicit diffusion (implicitness `θ`). The method is
second order for `θ = 1/2`.

The LU decomposition of the LHS matrix is computed every time the time step
changes.

Note that, in contrast to explicit methods, the pressure from previous time
steps has an influence on the accuracy of the velocity.
"""
struct AdamsBashforthCrankNicolsonMethod{T,M} <: AbstractODEMethod{T}
    α₁::T
    α₂::T
    θ::T
    p_add_solve::Bool
    method_startup::M
    AdamsBashforthCrankNicolsonMethod(
        T = Float64;
        α₁ = T(3 // 2),
        α₂ = T(-1 // 2),
        θ = T(1 // 2),
        p_add_solve = true,
        method_startup,
    ) = new{T,typeof(method_startup)}(α₁, α₂, θ, p_add_solve, method_startup)
end

"""
    OneLegMethod(
        T = Float64;
        β = T(1 // 2),
        p_add_solve = true,
        method_startup,
    )

Explicit one-leg β-method following symmetry-preserving discretization of
turbulent flow. See Verstappen and Veldman [Verstappen2003](@cite)
[Verstappen1997](@cite) for details.
"""
struct OneLegMethod{T,M} <: AbstractODEMethod{T}
    β::T
    p_add_solve::Bool
    method_startup::M
    OneLegMethod(T = Float64; β = T(1 // 2), p_add_solve = true, method_startup) =
        new{T,typeof(method_startup)}(β, p_add_solve, method_startup)
end

"""
    AbstractRungeKuttaMethod

Abstract Runge Kutta method.
"""
abstract type AbstractRungeKuttaMethod{T} <: AbstractODEMethod{T} end

"""
    ExplicitRungeKuttaMethod(; A, b, c, r, p_add_solve = true)

Explicit Runge Kutta method.
"""
Base.@kwdef struct ExplicitRungeKuttaMethod{T} <: AbstractRungeKuttaMethod{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    r::T
    p_add_solve::Bool = true
end

"""
    ImplicitRungeKuttaMethod(;
        A,
        b,
        c,
        r,
        newton_type = :full,
        maxiter = 10,
        abstol = 1e-14,
        reltol = 1e-14,
        p_add_solve = true,
    )

Implicit Runge Kutta method.

The implicit linear system is solved at each time step using Newton's method. The
`newton_type` may be one of the following:

- `:no`: Replace iteration matrix with I/Δt (no Jacobian)
- `:approximate`: Build Jacobian once before iterations only
- `:full`: Build Jacobian at each iteration
"""
Base.@kwdef struct ImplicitRungeKuttaMethod{T} <: AbstractRungeKuttaMethod{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    r::T
    newton_type::Symbol = :full
    maxiter::Int = 10
    abstol::T = 1e-14
    reltol::T = 1e-14
    p_add_solve::Bool = true
end

"""
    runge_kutta_method(A, b, c, r; [p_add_solve], [newton_type], [maxiter], [abstol], [reltol])

Get Runge Kutta method. The function checks whether the method is explicit.

`p_add_solve`: whether to add a pressure solve step to the method.

For implicit RK methods: `newton_type`, `maxiter`, `abstol`, `reltol`.
"""
function runge_kutta_method(A, b, c, r; T = Float64, kwargs...)
    s = size(A, 1)
    s == size(A, 2) == length(b) == length(c) ||
        error("A, b, and c must have the same sizes")
    isexplicit = all(≈(0), UpperTriangular(A))
    A = convert(Matrix{T}, A)
    b = convert(Vector{T}, b)
    c = convert(Vector{T}, c)
    r = convert(T, r)
    if isexplicit
        # Shift Butcher tableau, as A[1, :] is always zero for explicit methods
        A = [A[2:end, :]; b']
        # Vector with time instances (1 is the time level of final step)
        c = [c[2:end]; 1]
        ExplicitRungeKuttaMethod(; A, b, c, r, kwargs...)
    else
        ImplicitRungeKuttaMethod(; A, b, c, r, kwargs...)
    end
end
