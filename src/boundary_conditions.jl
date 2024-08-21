"Boundary condition for one side of the domain."
abstract type AbstractBC end

"Periodic boundary conditions. Must be periodic on both sides."
struct PeriodicBC <: AbstractBC end

"""
Dirichlet boundary conditions for the velocity, where `u[1] = (x..., t) ->
u1_BC` up to `u[d] = (x..., t) -> ud_BC`, where `d` is the dimension.

When `u` is `nothing`, then the boundary conditions are
no slip boundary conditions, where all velocity components are zero.

To make the pressure the same order as velocity, also provide `dudt`.
"""
struct DirichletBC{U,DUDT} <: AbstractBC
    "Boundary condition"
    u::U

    "Time derivative of boundary condition"
    dudt::DUDT

    DirichletBC(u = nothing, dudt = nothing) = new{typeof(u),typeof(dudt)}(u, dudt)
end

"""
Symmetric boundary conditions.
The parallel velocity and pressure is the same at each side of the boundary.
The normal velocity is zero.
"""
struct SymmetricBC <: AbstractBC end

"""
Pressure boundary conditions.
The pressure is prescribed on the boundary (usually an "outlet").
The velocity has zero Neumann conditions.

Note: Currently, the pressure is prescribed with the constant value of
zero on the entire boundary.
"""
struct PressureBC <: AbstractBC end

function ghost_a! end
function ghost_b! end

# Add opposite boundary ghost volume
# Do everything in first function call for periodic
function ghost_a!(::PeriodicBC, x)
    Δx_a = x[2] - x[1]
    Δx_b = x[end] - x[end-1]
    pushfirst!(x, x[1] - Δx_b)
    push!(x, x[end] + Δx_a)
end
ghost_b!(::PeriodicBC, x) = nothing

# Add infinitely thin boundary volume
ghost_a!(::DirichletBC, x) = pushfirst!(x, x[1])
ghost_b!(::DirichletBC, x) = push!(x, x[end])

# Duplicate boundary volume
ghost_a!(::SymmetricBC, x) = pushfirst!(x, x[1] - (x[2] - x[1]))
ghost_b!(::SymmetricBC, x) = push!(x, x[end] + (x[end] - x[end-1]))

# Add infinitely thin boundary volume
# On the left, we need to add two ghost volumes to have a normal component at
# the left of the first ghost volume
ghost_a!(::PressureBC, x) = pushfirst!(x, x[1], x[1])
ghost_b!(::PressureBC, x) = push!(x, x[end])

"""
    $FUNCTIONNAME(bc, isnormal, isright)

Number of non-DOF velocity components at boundary.
If `isnormal`, then the velocity is normal to the boundary, else parallel.
If `isright`, it is at the end/right/rear/top boundary, otherwise beginning.
"""
function offset_u end

"""
    $FUNCTIONNAME(bc, isnormal, isright)

Number of non-DOF pressure components at boundary.
"""
function offset_p end

offset_u(::PeriodicBC, isnormal, isright) = 1
offset_p(::PeriodicBC, isright) = 1

offset_u(::DirichletBC, isnormal, isright) = 1 + isnormal * isright
offset_p(::DirichletBC, isright) = 1

offset_u(::SymmetricBC, isnormal, isright) = 1 + isnormal * isright
offset_p(::SymmetricBC, isright) = 1

offset_u(::PressureBC, isnormal, isright) = 1 + !isnormal * !isright
offset_p(::PressureBC, isright) = 1 + !isright

function apply_bc_u! end
function apply_bc_p! end
function apply_bc_temp! end

apply_bc_u(u, t, setup; kwargs...) = apply_bc_u!(copy.(u), t, setup; kwargs...)
apply_bc_p(p, t, setup; kwargs...) = apply_bc_p!(copy(p), t, setup; kwargs...)
apply_bc_temp(temp, t, setup; kwargs...) = apply_bc_temp!(copy(temp), t, setup; kwargs...)

ChainRulesCore.rrule(::typeof(apply_bc_u), u, t, setup; kwargs...) = (
    apply_bc_u(u, t, setup; kwargs...),
    # With respect to (apply_bc_u, u, t, setup)
    φbar -> (
        NoTangent(),
        # Important: identity operator should be part of `apply_bc_u_pullback`,
        # but is actually implemented via the `copy` below instead.
        Tangent{typeof(u)}(apply_bc_u_pullback!(copy.((φbar...,)), t, setup; kwargs...)...),
        NoTangent(),
        NoTangent(),
    ),
)

ChainRulesCore.rrule(::typeof(apply_bc_p), p, t, setup) = (
    apply_bc_p(p, t, setup),
    # With respect to (apply_bc_p, p, t, setup)
    φbar -> (
        NoTangent(),
        apply_bc_p_pullback!(
            # Important: identity operator should be part of `apply_bc_p_pullback`,
            # but is actually implemented via the `copy` below instead.
            copy(unthunk(φbar)),
            t,
            setup,
        ),
        NoTangent(),
        NoTangent(),
    ),
)

ChainRulesCore.rrule(::typeof(apply_bc_temp), temp, t, setup) = (
    apply_bc_temp(temp, t, setup),
    # With respect to (apply_bc_temp, temp, t, setup)
    φbar -> (
        NoTangent(),
        apply_bc_temp_pullback!(
            # Important: identity operator should be part of `apply_bc_temp_pullback`,
            # but is actually implemented via the `copy` below instead.
            copy(unthunk(φbar)),
            t,
            setup,
        ),
        NoTangent(),
        NoTangent(),
    ),
)

function apply_bc_u!(u, t, setup; kwargs...)
    (; boundary_conditions) = setup
    D = length(u)
    for β = 1:D
        apply_bc_u!(boundary_conditions[β][1], u, β, t, setup; isright = false, kwargs...)
        apply_bc_u!(boundary_conditions[β][2], u, β, t, setup; isright = true, kwargs...)
    end
    u
end

function apply_bc_u_pullback!(φbar, t, setup; kwargs...)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_u_pullback!(
            boundary_conditions[β][1],
            φbar,
            β,
            t,
            setup;
            isright = false,
            kwargs...,
        )
        apply_bc_u_pullback!(
            boundary_conditions[β][2],
            φbar,
            β,
            t,
            setup;
            isright = true,
            kwargs...,
        )
    end
    φbar
end

function apply_bc_p!(p, t, setup; kwargs...)
    (; boundary_conditions, grid) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_p!(boundary_conditions[β][1], p, β, t, setup; isright = false)
        apply_bc_p!(boundary_conditions[β][2], p, β, t, setup; isright = true)
    end
    p
end

function apply_bc_p_pullback!(φbar, t, setup; kwargs...)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_p_pullback!(
            boundary_conditions[β][1],
            φbar,
            β,
            t,
            setup;
            isright = false,
            kwargs...,
        )
        apply_bc_p_pullback!(
            boundary_conditions[β][2],
            φbar,
            β,
            t,
            setup;
            isright = true,
            kwargs...,
        )
    end
    φbar
end

function apply_bc_temp!(temp, t, setup; kwargs...)
    (; temperature, grid) = setup
    (; boundary_conditions) = temperature
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_temp!(boundary_conditions[β][1], temp, β, t, setup; isright = false)
        apply_bc_temp!(boundary_conditions[β][2], temp, β, t, setup; isright = true)
    end
    temp
end

function apply_bc_temp_pullback!(φbar, t, setup; kwargs...)
    (; temperature, grid) = setup
    (; boundary_conditions) = temperature
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_temp_pullback!(
            boundary_conditions[β][1],
            φbar,
            β,
            t,
            setup;
            isright = false,
            kwargs...,
        )
        apply_bc_temp_pullback!(
            boundary_conditions[β][2],
            φbar,
            β,
            t,
            setup;
            isright = true,
            kwargs...,
        )
    end
    φbar
end

function apply_bc_u!(::PeriodicBC, u, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function _bc_a!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I] = u[α][I+(N[β]-2)*e(β)]
    end
    @kernel function _bc_b!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I+(N[β]-1)*e(β)] = u[α][I+e(β)]
    end
    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    for α = 1:D
        if isright
            _bc_b!(get_backend(u[1]), workgroupsize)(u, Val(α), Val(β); ndrange)
        else
            _bc_a!(get_backend(u[1]), workgroupsize)(u, Val(α), Val(β); ndrange)
        end
    end
    u
end

function apply_bc_u_pullback!(::PeriodicBC, φbar, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function adj_a!(φ, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        φ[α][I+(N[β]-2)*e(β)] += φ[α][I]
        φ[α][I] = 0
    end
    @kernel function adj_b!(φ, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        φ[α][I+e(β)] += φ[α][I+(N[β]-1)*e(β)]
        φ[α][I+(N[β]-1)*e(β)] = 0
    end
    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    for α = 1:D
        if isright
            adj_b!(get_backend(φbar[1]), workgroupsize)(φbar, Val(α), Val(β); ndrange)
        else
            adj_a!(get_backend(φbar[1]), workgroupsize)(φbar, Val(α), Val(β); ndrange)
        end
    end
    φbar
end

function apply_bc_p!(::PeriodicBC, p, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function _bc_a(p, ::Val{β}) where {β}
        I = @index(Global, Cartesian)
        p[I] = p[I+(N[β]-2)*e(β)]
    end
    @kernel function _bc_b(p, ::Val{β}) where {β}
        I = @index(Global, Cartesian)
        p[I+(N[β]-1)*e(β)] = p[I+e(β)]
    end
    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    if isright
        _bc_b(get_backend(p), workgroupsize)(p, Val(β); ndrange)
    else
        _bc_a(get_backend(p), workgroupsize)(p, Val(β); ndrange)
    end
    p
end

function apply_bc_p_pullback!(::PeriodicBC, φbar, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function adj_a!(φ, ::Val{β}) where {β}
        I = @index(Global, Cartesian)
        φ[I+(N[β]-2)*e(β)] += φ[I]
        φ[I] = 0
    end
    @kernel function adj_b!(φ, ::Val{β}) where {β}
        I = @index(Global, Cartesian)
        φ[I+e(β)] += φ[I+(N[β]-1)*e(β)]
        φ[I+(N[β]-1)*e(β)] = 0
    end
    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    if isright
        adj_b!(get_backend(φbar), workgroupsize)(φbar, Val(β); ndrange)
    else
        adj_a!(get_backend(φbar), workgroupsize)(φbar, Val(β); ndrange)
    end
    φbar
end

apply_bc_temp!(bc::PeriodicBC, temp, β, t, setup; isright, kwargs...) =
    apply_bc_p!(bc, temp, β, t, setup; isright, kwargs...)

apply_bc_temp_pullback!(bc::PeriodicBC, φbar, β, t, setup; isright, kwargs...) =
    apply_bc_p_pullback!(bc, φbar, β, t, setup; isright, kwargs...)

function apply_bc_u!(bc::DirichletBC, u, β, t, setup; isright, dudt = false, kwargs...)
    (; dimension, x, xp, N) = setup.grid
    D = dimension()
    e = Offset{D}()
    # isnothing(bc.u) && return
    bcfunc = dudt ? bc.dudt : bc.u
    for α = 1:D
        I = if isright
            CartesianIndices(
                ntuple(γ -> γ == β ? α == β ? (N[γ]-1:N[γ]-1) : (N[γ]:N[γ]) : (1:N[γ]), D),
            )
        else
            CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
        end
        xI = ntuple(
            γ -> reshape(
                γ == α ? x[γ][I.indices[α].+1] : xp[γ][I.indices[γ]],
                ntuple(Returns(1), γ - 1)...,
                :,
                ntuple(Returns(1), D - γ)...,
            ),
            D,
        )
        if isnothing(bc.u)
            u[α][I] .= 0
        else
            u[α][I] .= bcfunc.((Dimension(α),), xI..., t)
        end
    end
    u
end

# apply_bc_u_pullback!(::DirichletBC, φbar, β, t, setup; isright, kwargs...) =
#     @not_implemented("DirichletBC pullback not yet implemented.")

function apply_bc_p!(::DirichletBC, p, β, t, setup; isright, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    e = Offset{D}()
    if isright
        I = CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
        p[I] .= p[I.-e(β)]
    else
        I = CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
        p[I] .= p[I.+e(β)]
    end
    p
end

# apply_bc_p_pullback!(::DirichletBC, φbar, β, t, setup; isright, kwargs...) =
#     @not_implemented("DirichletBC pullback not yet implemented.")

function apply_bc_temp!(bc::DirichletBC, temp, β, t, setup; isright, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    e = Offset{D}()
    I = if isright
        CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
    else
        CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
    end
    temp[I] .= bc.u
    temp
end

function apply_bc_u!(::SymmetricBC, u, β, t, setup; isright, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    e = Offset{D}()
    for α = 1:D
        if α != β
            if isright
                I = CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
                u[α][I] .= u[α][I.-e(β)]
            else
                I = CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
                u[α][I] .= u[α][I.+e(β)]
            end
        end
    end
    u
end

# apply_bc_u_pullback!(::SymmetricBC, φbar, β, t, setup; isright, kwargs...) =
#     @not_implemented("SymmetricBC pullback not yet implemented.")

function apply_bc_p!(::SymmetricBC, p, β, t, setup; isright, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    e = Offset{D}()
    if isright
        I = CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
        p[I] .= p[I.-e(β)]
    else
        I = CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
        p[I] .= p[I.+e(β)]
    end
    p
end

# apply_bc_p_pullback!(::SymmetricBC, φbar, β, t, setup; isright, kwargs...) =
#     @not_implemented("SymmetricBC pullback not yet implemented.")

apply_bc_temp!(bc::SymmetricBC, temp, β, t, setup; isright, kwargs...) =
    apply_bc_p!(bc, temp, β, t, setup; isright, kwargs...)

function apply_bc_u!(bc::PressureBC, u, β, t, setup; isright, kwargs...)
    (; grid, workgroupsize) = setup
    (; dimension, N, Nu, Iu) = grid
    D = dimension()
    e = Offset{D}()
    @kernel function _bc_a!(u, ::Val{α}, ::Val{β}, I0) where {α,β}
        I = @index(Global, Cartesian)
        I = I + I0
        u[α][I] = u[α][I+e(β)]
    end
    @kernel function _bc_b!(u, ::Val{α}, ::Val{β}, I0) where {α,β}
        I = @index(Global, Cartesian)
        I = I + I0
        u[α][I] = u[α][I-e(β)]
    end
    ndrange = (N[1:β-1]..., 1, N[β+1:end]...)
    for α = 1:D
        if isright
            I0 = CartesianIndex(ntuple(γ -> γ == β ? N[β] : 1, D))
            I0 -= oneunit(I0)
            _bc_b!(get_backend(u[1]), workgroupsize)(u, Val(α), Val(β), I0; ndrange)
        else
            I0 = CartesianIndex(ntuple(γ -> γ == β && α != β ? 2 : 1, D))
            I0 -= oneunit(I0)
            _bc_a!(get_backend(u[1]), workgroupsize)(u, Val(α), Val(β), I0; ndrange)
        end
    end
    u
end

# apply_bc_u_pullback!(::PressureBC, φbar, β, t, setup; isright, kwargs...) =
#     @not_implemented("PressureBC pullback not yet implemented.")

function apply_bc_p!(bc::PressureBC, p, β, t, setup; isright, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    I = if isright
        CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
    else
        CartesianIndices(ntuple(γ -> γ == β ? (2:2) : (1:N[γ]), D))
    end
    p[I] .= 0
    p
end

apply_bc_p_pullback!(::PressureBC, φbar, β, t, setup; isright, kwargs...) =
    @not_implemented("PressureBC pullback not yet implemented.")
