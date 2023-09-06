"""
    AbstractViscosityModel

Abstract viscosity model.
"""
abstract type AbstractViscosityModel end

"""
    LaminarModel()

Laminar model.
"""
Base.@kwdef struct LaminarModel <: AbstractViscosityModel
end

"""
    MixingLengthModel()

Mixing-length model with mixing length `lm`.
"""
Base.@kwdef struct MixingLengthModel{T} <: AbstractViscosityModel
    lm::T = 1 # Mixing length
end

"""
    SmagorinskyModel(Re, C_s = 0.17)

Smagorinsky-Lilly model with constant `C_s`.
"""
Base.@kwdef struct SmagorinskyModel{T} <: AbstractViscosityModel
    C_s::T = 0.17 # Smagorinsky constant
end

"""
    QR(Re)

QR-model.
"""
struct QRModel{T} <: AbstractViscosityModel
end
