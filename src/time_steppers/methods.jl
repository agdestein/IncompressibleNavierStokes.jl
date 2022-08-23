"""
    AbstractODEMethod

Abstract ODE method.
"""
abstract type AbstractODEMethod{T} end

"""
    AdamsBashforthCrankNicolsonMethod(; α₁ = 3 // 2, α₂ = -1 // 2, θ = 1 // 2, p_add_solve = true)

IMEX AB-CN: Adams-Bashforth for explicit convection (parameters `α₁` and `α₂`) and
Crank-Nicolson for implicit diffusion (implicitness `θ`).
The method is second order for `θ = 1/2`.

Adams-Bashforth for convection and Crank-Nicolson for diffusion
formulation:

```math
\\begin{align*}
(\\mathbf{u}^{n+1} - \\mathbf{u}^n) / Δt & =
    -(\\alpha_1 \\mathbf{c}^n + \\alpha_2 \\mathbf{c}^{n-1}) \\\\
    & + \\theta \\mathbf{d}^{n+1} + (1-\\theta) \\mathbf{d}^n \\\\
    & + \\theta \\mathbf{F}^{n+1} + (1-\\theta) \\mathbf{F}^n \\\\
    & + \\theta \\mathbf{BC}^{n+1} + (1-\\theta) \\mathbf{BC}^n \\\\
    & - \\mathbf{G} \\mathbf{p} + \\mathbf{y}_p
\\end{align*}
```

where BC are boundary conditions of diffusion. This is rewritten as:

```math
\\begin{align*}
(\\frac{1}{\\Delta t} \\mathbf{I} - \\theta \\mathbf{D}) \\mathbf{u}^{n+1} & =
    (\\frac{1}{\\Delta t} \\mathbf{I} - (1 - \\theta) \\mathbf{D}) \\mathbf{u}^{n} \\\\
    & - (\\alpha_1 \\mathbf{c}^n + \\alpha_2 \\mathbf{c}^{n-1}) \\\\
    & + \\theta \\mathbf{F}^{n+1} + (1-\\theta) \\mathbf{F}^{n} \\\\
    & + \\theta \\mathbf{BC}^{n+1} + (1-\\theta) \\mathbf{BC}^{n} \\\\
    & - \\mathbf{G} \\mathbf{p} + \\mathbf{y}_p
\\end{align*}
```

The LU decomposition of the LHS matrix is precomputed in `operator_convection_diffusion.jl`.

Note that, in constrast to explicit methods, the pressure from previous time steps has an
influence on the accuracy of the velocity.

"""
Base.@kwdef struct AdamsBashforthCrankNicolsonMethod{T} <: AbstractODEMethod{T}
    α₁::T = 3 // 2
    α₂::T = -1 // 2
    θ::T = 1 // 2
    p_add_solve::Bool = true
end

"""
    OneLegMethod(; β = 1 // 2, p_add_solve = true)

Explicit one-leg β-method following symmetry-preserving discretization of
turbulent flow. See [Verstappen and Veldman (JCP 2003)] for details, or [Direct numerical
simulation of turbulence at lower costs (Journal of Engineering Mathematics 1997)].

Formulation:

```math
\\frac{(\\beta + 1/2) u^{n+1} - 2 \\beta u^{n} + (\\beta - 1/2) u^{n-1}}{\\Delta t} = F((1 +
\\beta) u^n - \\beta u^{n-1}).
```
"""
Base.@kwdef struct OneLegMethod{T} <: AbstractODEMethod{T}
    β::T = 1 // 2
    p_add_solve::Bool = true
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
function runge_kutta_method(A, b, c, r; kwargs...)
    s = size(A, 1)
    s == size(A, 2) == length(b) == length(c) ||
        error("A, b, and c must have the same sizes")
    isexplicit = all(≈(0), UpperTriangular(A))
    # T = promote_type(eltype(A), eltype(b), eltype(c), typeof(r))
    # TODO: Find where to pass T
    T = Float64
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

