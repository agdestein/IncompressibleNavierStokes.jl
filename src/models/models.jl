abstract type AbstractViscosityModel{T} end

struct LaminarModel{T} <: AbstractViscosityModel{T} end
struct KEpsilonModel{T} <: AbstractViscosityModel{T} end
Base.@kwdef struct MixingLengthModel{T} <: AbstractViscosityModel{T}
    lm::T = 1 # Mixing length
end
Base.@kwdef struct SmagorinskyModel{T} <: AbstractViscosityModel{T}
    C_s::T = 0.17 # Smagorinsky constant
end
struct QRModel{T} <: AbstractViscosityModel{T} end
