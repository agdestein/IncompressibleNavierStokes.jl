"""
Abstract ODE method.
"""
abstract type AbstractODEMethod{T} end

@doc raw"""
Explicit one-leg β-method following symmetry-preserving discretization of
turbulent flow. See Verstappen and Veldman [Verstappen2003](@cite)
[Verstappen1997](@cite) for details.

We here require that the time step ``\Delta t`` is constant. Given the velocity
``u_0`` and pressure ``p_0`` at the current time ``t_0`` and their previous
values ``u_{-1}`` and ``p_{-1}`` at the time ``t_{-1} = t_0 - \Delta t``, we
start by computing the "offstep" values ``v = (1 + \beta) v_0 - \beta v_{-1}``
and ``Q = (1 + \beta) p_0 - \beta p_{-1}`` for some ``\beta = \frac{1}{2}``.

A tentative velocity field ``\tilde{v}`` is then computed as follows:

```math
\tilde{v} = \frac{1}{\beta + \frac{1}{2}} \left( 2 \beta u_0 - \left( \beta -
\frac{1}{2} \right) u_{-1} + \Delta t F(v, t) - \Delta t
(G Q + y_G(t)) \right).
```

A pressure correction ``\Delta p `` is obtained by solving the Poisson equation
```math
L \Delta p = \frac{\beta + \frac{1}{2}}{\Delta t} W (M \tilde{v} + y_M(t)).
```

Finally, the divergence free velocity field is given by

```math
u = \tilde{v} - \frac{\Delta t}{\beta + \frac{1}{2}} G \Delta p,
```

while the second order accurate pressure is given by

```math
p = 2 p_0 - p_{-1} + \frac{4}{3} \Delta p.
```
"""
struct OneLegMethod{T,M} <: AbstractODEMethod{T}
    β::T
    p_add_solve::Bool
    method_startup::M
    OneLegMethod(T = Float64; β = T(1 // 2), p_add_solve = true, method_startup) =
        new{T,typeof(method_startup)}(β, p_add_solve, method_startup)
end

"""
Abstract Runge Kutta method.
"""
abstract type AbstractRungeKuttaMethod{T} <: AbstractODEMethod{T} end

@doc raw"""
Explicit Runge Kutta method.
See Sanderse [Sanderse2012](@cite).

Consider the velocity field ``u_0`` at a certain time ``t_0``. We will now
perform one time step to ``t = t_0 + \Delta t``. For explicit Runge-Kutta
methods, this time step is divided into ``s`` sub-steps ``t_i = t_0 + \Delta
t_i`` with increment ``\Delta t_i = c_i \Delta t``. The final substep performs
the full time step ``\Delta t_s = \Delta t`` such that ``t_s = t``.

For ``i = 1, \dots, s``, the intermediate velocity ``u_i`` and pressure ``p_i``
are computed as follows:

```math
\begin{split}
k_i & = F(u_{i - 1}, t_{i - 1}) - y_G(t_{i - 1}) \\
v_i & = u_0 + \Delta t \sum_{j = 1}^i a_{i j} k_j \\
L p_i & = W M \frac{1}{c_i} \sum_{j = 1}^i a_{i j} k_j +
W \frac{y_M(t_i) - y_M(t_0)}{\Delta t_i} \\
& = W \frac{(M v_i + y_M(t_i)) - (M u_0 + y_M(t_0))}{\Delta t_i^n} \\
& = W \frac{M v_i + y_M(t_i)}{\Delta t_i^n} \\
u_i & = v_i - \Delta t_i G p_i,
\end{split}
```

where ``(a_{i j})_{i j}`` are the Butcher tableau coefficients of the
RK-method, with the convention ``c_i = \sum_{j = 1}^i a_{i j}``.

Finally, we return ``u_s``. If ``u_0 = u(t_0)``, we get the accuracy ``u_s =
u(t) + \mathcal{O}(\Delta t^{r + 1})``, where ``r`` is the order of the
RK-method. If we perform ``n`` RK time steps instead of one, starting at exact
initial conditions ``u^0 = u(0)``, then ``u^n = u(t^n) + \mathcal{O}(\Delta
t^r)`` for all ``n \in \{1, \dots, N\}``. Note that for a given ``u``, the
corresponding pressure ``p`` can be calculated to the same accuracy as ``u`` by
doing an additional pressure projection after each outer time step ``\Delta t``
(if we know ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t)``), or to first order
accuracy by simply returning ``p_s``.

Note that each of the sub-step velocities ``u_i`` is divergence free, after
projecting the tentative velocities ``v_i``. This is ensured due to the
judiciously chosen replacement of ``\frac{\mathrm{d} y_M}{\mathrm{d} t}(t_i)``
with ``(y_M(t_i) - y_M(t_0)) / \Delta t_i``. The space-discrete
divergence-freeness is thus perfectly preserved, even though the time
discretization introduces other errors.
"""
Base.@kwdef struct ExplicitRungeKuttaMethod{T} <: AbstractRungeKuttaMethod{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    r::T
    p_add_solve::Bool = true
end

"""
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
        error("Only explicit Runge-Kutta methods are supported")
    end
end

"""
Low memory Wray 3rd order scheme.
Uses 3 vector fields and one scalar field.
"""
struct LMWray3{T} <: AbstractRungeKuttaMethod{T}
    LMWray3(; T = Float64) = new{T}()
end
