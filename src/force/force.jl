"""
    Force{T}

Forcing parameters with floating point type `T`.
"""
Base.@kwdef mutable struct Force{T}
    x_c::T = 0                                               # X-coordinate of body
    y_c::T = 0                                               # Y-coordinate of body
    Ct::T = 0                                                # Thrust coefficient for actuator disk computations
    D::T = 1                                                 # Actuator disk diameter
    F::Vector{T} = T[]                                       # For storing constant body force
    isforce::Bool = false                                    # Presence of a force file
    force_unsteady::Bool = false                             # Unsteady forcing 
    bodyforce_x::Function = () -> error("bodyforce_x not implemented")
    bodyforce_y::Function = () -> error("bodyforce_y not implemented")
    Fp::Function = () -> error("Fp not implemented")
end
