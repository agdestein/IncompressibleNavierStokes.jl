abstract type AbstractBC end

"""
    PeriodicBC()

Periodic boundary conditions. Must be periodic on both sides.
"""
struct PeriodicBC <: AbstractBC end

"""
    DirichletBC()

No slip boundary conditions, where all velocity components are zero.

    DirichletBC(u, dudt)

Dirichlet boundary conditions for the velocity, where `u[1] = (x..., t) ->
u1_BC` up to `u[d] = (x..., t) -> ud_BC`, where `d` is the dimension.

To make the pressure the same order as velocity, also provide `dudt`.
"""
struct DirichletBC{F,G} <: AbstractBC
    u::F
    dudt::G
end

DirichletBC() = DirichletBC(nothing, nothing)

"""
    SymmetricBC()

Symmetric boundary conditions.
The parallel velocity and pressure is the same at each side of the boundary.
The normal velocity is zero.
"""
struct SymmetricBC <: AbstractBC end

"""
    PressureBC()

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
ghost_a!(::SymmetricBC, x) = pushfirst!(x, x[1] + (x[2] - x[1]))
ghost_b!(::SymmetricBC, x) = push!(x, x[end] + (x[end] - x[end-1]))

# Add infinitely thin boundary volume
# On the left, we need to add two ghost volumes to have a normal component at
# the left of the first ghost volume
ghost_a!(::PressureBC, x) = pushfirst!(x, x[1], x[1])
ghost_b!(::PressureBC, x) = push!(x, x[end])

"""
    offset_u(bc, isnormal, atend)

Number of non-DOF velocity components at boundary.
If `isnormal`, then the velocity is normal to the boundary, else parallel.
If `atend`, it is at the end/right/rear/top boundary, otherwise beginning.
"""
function offset_u end

"""
    offset_p(bc)

Number of non-DOF pressure components at boundary.
"""
function offset_p end

offset_u(::PeriodicBC, isnormal, atend) = 1
offset_p(::PeriodicBC, atend) = 1

offset_u(::DirichletBC, isnormal, atend) = 1 + isnormal * atend
offset_p(::DirichletBC, atend) = 1

offset_u(::SymmetricBC, isnormal, atend) = 1 + isnormal * atend
offset_p(::SymmetricBC, atend) = 1

offset_u(::PressureBC, isnormal, atend) = 1 + !isnormal * !atend
offset_p(::PressureBC, atend) = 1 + !atend

function apply_bc_u! end
function apply_bc_p! end

function apply_bc_u!(u, t, setup; kwargs...)
    (; boundary_conditions) = setup
    D = length(u)
    for β = 1:D
        apply_bc_u!(boundary_conditions[β][1], u, β, t, setup; atend = false, kwargs...)
        apply_bc_u!(boundary_conditions[β][2], u, β, t, setup; atend = true, kwargs...)
    end
end

function apply_bc_p!(p, t, setup; kwargs...)
    (; boundary_conditions, grid) = setup
    (; dimension) = grid
    D = dimension()
    for β = 1:D
        apply_bc_p!(boundary_conditions[β][1], p, β, t, setup; atend = false)
        apply_bc_p!(boundary_conditions[β][2], p, β, t, setup; atend = true)
    end
end

function apply_bc_u!(::PeriodicBC, u, β, t, setup; atend, kwargs...)
    (; grid) = setup
    (; dimension, N) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bc_a!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I] = u[α][I+(N[β]-2)*δ(β)]
    end
    @kernel function _bc_b!(u, ::Val{α}, ::Val{β}) where {α,β}
        I = @index(Global, Cartesian)
        u[α][I+(N[β]-1)*δ(β)] = u[α][I+δ(β)]
    end
    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    for α = 1:D
        if atend
            _bc_b!(get_backend(u[1]), WORKGROUP)(u, Val(α), Val(β); ndrange)
        else
            _bc_a!(get_backend(u[1]), WORKGROUP)(u, Val(α), Val(β); ndrange)
        end
        synchronize(get_backend(u[1]))
    end
end

function apply_bc_p!(::PeriodicBC, p, β, t, setup; atend, kwargs...)
    (; grid) = setup
    (; dimension, N) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bc_a(p, ::Val{β}) where {β}
        I = @index(Global, Cartesian)
        p[I] = p[I+(N[β]-2)*δ(β)]
    end
    @kernel function _bc_b(p, ::Val{β}) where {β}
        I = @index(Global, Cartesian)
        p[I+(N[β]-1)*δ(β)] = p[I+δ(β)]
    end
    ndrange = ntuple(γ -> γ == β ? 1 : N[γ], D)
    if atend
        _bc_b(get_backend(p), WORKGROUP)(p, Val(β); ndrange)
    else
        _bc_a(get_backend(p), WORKGROUP)(p, Val(β); ndrange)
    end
end

function apply_bc_u!(bc::DirichletBC, u, β, t, setup; atend, dudt = false, kwargs...)
    (; dimension, x, xp, N) = setup.grid
    D = dimension()
    δ = Offset{D}()
    isnothing(bc.u) && return
    bcfunc = dudt ? bc.dudt : bc.u
    for α = 1:D
        if atend
            I = CartesianIndices(
                ntuple(γ -> γ == β ? α == β ? (N[γ]-1:N[γ]-1) : (N[γ]:N[γ]) : (1:N[γ]), D),
            )
        else
            I = CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
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
        u[α][I] .= bcfunc.(Val(α), xI..., t)
    end
end

function apply_bc_p!(::DirichletBC, p, β, t, setup; atend, kwargs...)
    nothing
end

function apply_bc_u!(::SymmetricBC, u, β, t, setup; atend, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    δ = Offset{D}()
    for α = 1:D
        if α != β
            if atend
                I = CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
                u[α][I] .= u[α][I.-δ(β)]
            else
                I = CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
                u[α][I] .= u[α][I.+δ(β)]
            end
        end
    end
end

function apply_bc_p!(::SymmetricBC, p, β, t, setup; atend, kwargs...)
    (; dimension, N) = setup.grid
    D = dimension()
    δ = Offset{D}()
    if atend
        I = CartesianIndices(ntuple(γ -> γ == β ? (N[γ]:N[γ]) : (1:N[γ]), D))
        p[I] .= p[I.-δ(β)]
    else
        I = CartesianIndices(ntuple(γ -> γ == β ? (1:1) : (1:N[γ]), D))
        p[I] .= p[I.+δ(β)]
    end
end

function apply_bc_u!(bc::PressureBC, u, β, t, setup; atend, kwargs...)
    (; grid) = setup
    (; dimension, N, Nu, Iu) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bc_a!(u, ::Val{α}, ::Val{β}, I0) where {α,β}
        I = @index(Global, Cartesian)
        I = I + I0
        u[α][I] = u[α][I+δ(β)]
    end
    @kernel function _bc_b!(u, ::Val{α}, ::Val{β}, I0) where {α,β}
        I = @index(Global, Cartesian)
        I = I + I0
        u[α][I] = u[α][I-δ(β)]
    end
    ndrange = (N[1:β-1]..., 1, N[β+1:end]...)
    for α = 1:D
        if atend
            I0 = CartesianIndex(ntuple(γ -> γ == β ? N[β] : 1, D))
            I0 -= oneunit(I0)
            _bc_b!(get_backend(u[1]), WORKGROUP)(u, Val(α), Val(β), I0; ndrange)
        else
            I0 = CartesianIndex(ntuple(γ -> γ == β && α != β ? 2 : 1, D))
            I0 -= oneunit(I0)
            _bc_a!(get_backend(u[1]), WORKGROUP)(u, Val(α), Val(β), I0; ndrange)
        end
    end
end

function apply_bc_p!(bc::PressureBC, p, β, t, setup; atend, kwargs...)
    # p is already zero at boundary
    nothing
end
