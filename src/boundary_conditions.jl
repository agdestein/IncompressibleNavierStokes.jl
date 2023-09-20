abstract type AbstractBC end

struct PeriodicBC <: AbstractBC end

"""
    DirichletBC()

No split boundary conditions, where all velocity components are zero.

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

# Add infinitely thin boundary volume and then symmetric
ghost_a!(::PressureBC, x) = pushfirst!(x, x[1] - (x[2] - x[1]), x[1])
ghost_b!(::PressureBC, x) = push!(x, x[end], x[end] + (x[end] - x[end-1]))

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
offset_p(::PeriodicBC) = 1

offset_u(::DirichletBC, isnormal, atend) = 1
offset_p(::DirichletBC) = 1

offset_u(::SymmetricBC, isnormal, atend) = 1
offset_p(::SymmetricBC) = 1

offset_u(::PressureBC, isnormal, atend) = isnormal && atend ? 2 : 1
offset_p(::PressureBC) = 2

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

function apply_bc_p!(p, t, setup)
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
    (; dimension, Nu, Iu) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bc_a!(u, α, β, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        u[α][I] = u[α][I+Nu[α][β]*δ(β)]
    end
    @kernel function _bc_b!(u, α, β, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        u[α][I+Nu[α][β]*δ(β)] = u[α][I]
    end
    for α = 1:D
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        ndrange = (Nu[α][1:β-1]..., 1, Nu[α][β+1:end]...)
        if atend
            _bc_b!(get_backend(u[1]), WORKGROUP)(u, α, β, I0; ndrange)
            synchronize(get_backend(u[1]))
        else
            _bc_a!(get_backend(u[1]), WORKGROUP)(u, α, β, I0 - δ(β); ndrange)
            synchronize(get_backend(u[1]))
        end
    end
end

function apply_bc_p!(::PeriodicBC, p, β, t, setup; atend)
    (; grid) = setup
    (; dimension, Np, Ip) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _bc_a(p, β, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        p[I] = p[I+Np[β]*δ(β)]
    end
    @kernel function _bc_b(p, β, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        p[I+Np[β]*δ(β)] = p[I]
    end
    I0 = first(Ip)
    I0 -= oneunit(I0)
    ndrange = (Np[1:β-1]..., 1, Np[β+1:end]...)
    if atend
        _bc_b(get_backend(p), WORKGROUP)(p, β, I0; ndrange)
        synchronize(get_backend(p))
    else
        _bc_a(get_backend(p), WORKGROUP)(p, β, I0 - δ(β); ndrange)
        synchronize(get_backend(p))
    end
end

function apply_bc_u!(bc::DirichletBC, u, β, t, setup; atend, dudt = false, kwargs...)
    (; dimension, Nu, x, xp) = setup.grid
    D = dimension()
    δ = Offset{D}()
    bcfunc = dudt ? bc.dudt : bc.u
    @kernel function _bc_a(u, α, β)
        I = @index(Global, Cartesian)
        # u[i][I] = bcfunc[i](ntuple(k -> k == i [I + Nu[i][j] * δ(j)]
        # TODO: Apply bcfunc
    end
    @kernel function _bc_b(u, α, β; xΓ)
        I = @index(Global, Cartesian)
        # u[α][I] = bcfunc[α](ntuple(xp)
        # TODO: Apply bcfunc
    end
    for α = 1:D
        xΓ = (xp[1:β-1]..., xp[β+1:end]...)
        if atend
            _bc_b(get_backend(u[1]), WORKGROUP)(u, α, β, I0)
            synchronize(get_backend(u[1]))
        else
            _bc_a(get_backend(u[1]), WORKGROUP)(u, α, β, I0)
            synchronize(get_backend(u[1]))
        end
    end
end

function apply_bc_p!(::DirichletBC, p, β, t, setup; atend)
    nothing
end

function apply_bc_u!(::SymmetricBC, u, β, t, setup; atend, kwargs...)
    error("Not implemented")
    (; Nu, x, xp) = setup.grid
    D = dimension()
    δ = Offset{D}()
    bcfunc = dudt ? bc.dudt : bc.u
    @kernel function _bc_a(u, α, β)
        I = @index(Global, Cartesian)
        # u[i][I] = bcfunc[i](ntuple(k -> k == i [I + Nu[i][j] * δ(j)]
        # TODO: Apply bcfunc
    end
    @kernel function _bc_b(u, α, β)
        I = @index(Global, Cartesian)
        # u[α][I] = bcfunc[α](ntuple(xp)
        # TODO: Apply bcfunc
    end
    for α = 1:D
        for β = 1:D
            xΓ = (xp[1:β-1]..., xp[β+1:end]...)
            if atend
                _bc_b(get_backend(u[1]), WORKGROUP)(u, α, β, I0)
                synchronize(get_backend(u[1]))
            else
                _bc_a(get_backend(u[1]), WORKGROUP)(u, α, β, I0)
                synchronize(get_backend(u[1]))
            end
        end
    end
end

function apply_bc_p!(::SymmetricBC, p, β, t, setup; atend)
    error("Not implemented")
end

function apply_bc_u!(bc::PressureBC, u, β, t, setup; atend, kwargs...)
    error("Not implemented")
end

function apply_bc_p!(bc::PressureBC, p, β, t, setup; atend)
    error("Not implemented")
end
