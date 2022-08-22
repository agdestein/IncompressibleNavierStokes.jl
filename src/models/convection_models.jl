abstract type AbstractConvectionModel end

"""
    NoRegConvection()

Unregularized convection model.
"""
struct NoRegConvectionModel <: AbstractConvectionModel end

"""
    C2ConvectionModel()

C2 regularization convection model.
"""
struct C2ConvectionModel <: AbstractConvectionModel end

"""
    C4ConvectionModel()

C4 regularization convection model.
"""
struct C4ConvectionModel <: AbstractConvectionModel end

"""
    LerayConvectionModel()

Leray regularization convection model.
"""
struct LerayConvectionModel <: AbstractConvectionModel end
