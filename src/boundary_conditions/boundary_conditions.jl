"""
    BoundaryConditions{T}

Boundary conditions with floating point type `T`.
"""
Base.@kwdef mutable struct BoundaryConditions{T}
    bc_unsteady::Bool = false
    u::NamedTuple = (;)
    v::NamedTuple = (;)
    w::NamedTuple = (;)
    k::NamedTuple = (;)
    e::NamedTuple = (;)
    ν::NamedTuple = (;)
    u_bc::Function = (args...) -> error("u_bc not implemented")
    v_bc::Function = (args...) -> error("v_bc not implemented")
    w_bc::Function = (args...) -> error("w_bc not implemented")
    dudt_bc::Function = (args...) -> error("dudt_bc not implemented")
    dvdt_bc::Function = (args...) -> error("dvdt_bc not implemented")
    dwdt_bc::Function = (args...) -> error("dwdt_bc not implemented")
    p_bc::NamedTuple = (;)
    k_bc::NamedTuple = (;)
    e_bc::NamedTuple = (;)
end

"""
    BoundaryConditions(u_bc, v_bc; T = Float64, bc_unsteady, bc_type, kwargs...)

Create discrete boundary condtions.

Values should either be scalars or vectors. All values `(u, v, p, k, e)` are
defined at (x, y) locations, i.e. the corners of pressure volumes, so they
cover the entire domain, including corners.
"""
function BoundaryConditions(u_bc, v_bc; T = Float64, bc_unsteady, bc_type, kwargs...)
    bc_type.u.x[1] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for u-left")
    bc_type.u.x[2] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for u-right")
    bc_type.u.y[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for u-low")
    bc_type.u.y[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for u-up")

    bc_type.v.x[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for v-left")
    bc_type.v.x[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for v-right")
    bc_type.v.y[1] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for v-low")
    bc_type.v.y[2] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for v-up")

    # Pressure (for boundaries marked with `:pressure`)
    p∞ = zero(T)
    p_bc = (; x = (p∞, p∞), y = (p∞, p∞))

    # K-eps values
    k_bc = (; x = (zero(T), zero(T)), y = (zero(T), zero(T)))
    e_bc = (; x = (zero(T), zero(T)), y = (zero(T), zero(T)))

    BoundaryConditions{T}(;
        bc_unsteady,
        bc_type...,
        u_bc,
        v_bc,
        p_bc,
        k_bc,
        e_bc,
        kwargs...,
    )
end

"""
    BoundaryConditions(u_bc, v_bc, w_bc; T = Float64, bc_unsteady, bc_type, kwargs...)

Create discrete boundary condtions.

Values should either be scalars or vectors. All values `(u, v, p, k, e)` are
defined at (x, y, z) locations, i.e. the corners of pressure volumes, so they
cover the entire domain, including corners.
"""
function BoundaryConditions(u_bc, v_bc, w_bc; T = Float64, bc_unsteady, bc_type, kwargs...)
    bc_type.u.x[1] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for u-left")
    bc_type.u.x[2] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for u-right")
    bc_type.u.y[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for u-low")
    bc_type.u.y[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for u-up")
    bc_type.u.z[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for u-back")
    bc_type.u.z[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for u-front")

    bc_type.v.x[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for v-left")
    bc_type.v.x[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for v-right")
    bc_type.v.y[1] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for v-low")
    bc_type.v.y[2] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for v-up")
    bc_type.v.z[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for v-back")
    bc_type.v.z[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for v-front")

    bc_type.w.x[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for w-left")
    bc_type.w.x[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for w-right")
    bc_type.w.y[1] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for w-low")
    bc_type.w.y[2] ∈ (:dirichlet, :periodic, :symmetric) || error("Wrong BC for w-up")
    bc_type.w.z[1] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for w-back")
    bc_type.w.z[2] ∈ (:dirichlet, :periodic, :pressure) || error("Wrong BC for w-front")

    # Pressure (for boundaries marked with `:pressure`)
    p∞ = zero(T)
    p_bc = (; x = (p∞, p∞), y = (p∞, p∞), z = (p∞, p∞))

    # K-eps values
    k_bc = (; x = (zero(T), zero(T)), y = (zero(T), zero(T)), z = (zero(T), zero(T)))
    e_bc = (; x = (zero(T), zero(T)), y = (zero(T), zero(T)), z = (zero(T), zero(T)))

    BoundaryConditions{T}(;
        bc_unsteady,
        bc_type...,
        u_bc,
        v_bc,
        w_bc,
        p_bc,
        k_bc,
        e_bc,
        kwargs...,
    )
end
