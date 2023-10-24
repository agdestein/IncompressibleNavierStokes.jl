"""
    AbstractViscosityModel

Abstract viscosity model.
"""
abstract type AbstractViscosityModel end

"""
    LaminarModel()

Laminar model. This model assumes that there are no
sub-grid stresses. It can be used if the grid is sufficiently refined for the
given flow. It has the advantage of having a constant diffusion operator.
"""
struct LaminarModel <: AbstractViscosityModel
end

"""
    MixingLengthModel()

Mixing-length model with mixing length `lm`.
"""
@kwdef struct MixingLengthModel{T} <: AbstractViscosityModel
    lm::T = 1 # Mixing length
end

"""
    SmagorinskyModel(C_s = 0.17)

Smagorinsky-Lilly model with constant `C_s`.
"""
@kwdef struct SmagorinskyModel{T} <: AbstractViscosityModel
    C_s::T = 0.17 # Smagorinsky constant
end

"""
    QR(Re)

QR-model.
"""
struct QRModel{T} <: AbstractViscosityModel
end
