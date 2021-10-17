abstract type ViscosityModel{T} end

struct LaminarModel{T} <: ViscosityModel{T} end
struct KEpsilonModel{T} <: ViscosityModel{T} end
struct MixingLengthModel{T} <: ViscosityModel{T}
    lm::T = 1 # Mixing length
end
struct SmagorinskyModel{T} <: ViscosityModel{T}
    Cs::T = 0.17 # Smagorinsky constant
end
struct QRModel{T} <: ViscosityModel{T}
