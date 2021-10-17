abstract type ViscosityModel{T} end

struct LaminarModel{T} <: ViscosityModel{T} end
struct KEpsilonModel{T} <: ViscosityModel{T} end
Base.@kwdef struct MixingLengthModel{T} <: ViscosityModel{T}
    lm::T = 1 # Mixing length
end
Base.@kwdef struct SmagorinskyModel{T} <: ViscosityModel{T}
    Cs::T = 0.17 # Smagorinsky constant
end
struct QRModel{T} <: ViscosityModel{T} end
