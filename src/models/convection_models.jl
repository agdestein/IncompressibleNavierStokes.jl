abstract type AbstractConvectionModel{T} end

"""
    NoRegConvection()

Unregularized convection model.
"""
struct NoRegConvectionModel{T} <: AbstractConvectionModel{T} end

"""
    C2ConvectionModel()

C2 regularization convection model.
"""
struct C2ConvectionModel{T} <: AbstractConvectionModel{T} end

"""
    C4ConvectionModel()

C4 regularization convection model.
"""
struct C4ConvectionModel{T} <: AbstractConvectionModel{T} end

"""
    LerayConvectionModel()

Leray regularization convection model.
"""
struct LerayConvectionModel{T} <: AbstractConvectionModel{T} end
