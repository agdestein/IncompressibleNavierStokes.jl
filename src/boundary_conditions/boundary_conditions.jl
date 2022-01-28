"""
    BC{T}

Boundary conditions with floating point type `T`.
"""
Base.@kwdef mutable struct BC{T}
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
