"""
    BC{T}

Boundary conditions with floating point type `T`.
"""
Base.@kwdef mutable struct BC{T}
    bc_unsteady::Bool = false
    u::NamedTuple = (;)
    v::NamedTuple = (;)
    k::NamedTuple = (;)
    e::NamedTuple = (;)
    ν::NamedTuple = (;)
    u_bc::Function = () -> error("u_bc not implemented")
    v_bc::Function = () -> error("v_bc not implemented")
    dudt_bc::Function = () -> error("dudt_bc not implemented")
    dvdt_bc::Function = () -> error("dvdt_bc not implemented")
    p_bc::NamedTuple = (;)
    k_bc::NamedTuple = (;)
    e_bc::NamedTuple = (;)
    bc_type::Function = () -> error("bc_type not implemented")
end
