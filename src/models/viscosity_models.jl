"""
    AbstractViscosityModel

Abstract viscosity model.
"""
abstract type AbstractViscosityModel{T} end

"""
    LaminarModel(Re)

Laminar model with Reynolds number `Re`.
"""
Base.@kwdef struct LaminarModel{T} <: AbstractViscosityModel{T}
    Re::T # Reynolds number
end

"""
    MixingLengthModel(Re)

Mixing-length model with Reynolds number `Re` and mixing length `lm`.
"""
Base.@kwdef struct MixingLengthModel{T} <: AbstractViscosityModel{T}
    Re::T # Reynolds number
    lm::T = 1 # Mixing length
end

"""
    SmagorinskyModel(Re, C_s = 0.17)

Smagorinsky-Lilly model with Reynolds number `Re` and constant `C_s`.
"""
Base.@kwdef struct SmagorinskyModel{T} <: AbstractViscosityModel{T}
    Re::T # Reynolds number
    C_s::T = 0.17 # Smagorinsky constant
end

"""
    QR(Re)

QR-model with Reynolds number `Re`.
"""
Base.@kwdef struct QRModel{T} <: AbstractViscosityModel{T}
    Re::T # Reynolds number
end
