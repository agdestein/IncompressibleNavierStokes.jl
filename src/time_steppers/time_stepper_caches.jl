"""
    TimeStepperCache

Time stepper cache.
"""
abstract type TimeStepperCache end

struct ExplicitRungeKuttaStepperCache{T} <: TimeStepperCache
    kV::Matrix{T}
    kp::Matrix{T}
    Vtemp::Vector{T}
    Vtemp2::Vector{T}
    F::Vector{T}
    ∇F::SparseMatrixCSC{T, Int}
    f::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

struct ImplicitRungeKuttaStepperCache <: TimeStepperCache end
struct AdamsBashforthCrankNicolsonStepperCache <: TimeStepperCache end
struct OneLegStepperCache <: TimeStepperCache end

"""
    time_stepper_cache(stepper, args...; kwargs...)

Get time stepper cache for the given time stepper.
"""
function time_stepper_cache end

function time_stepper_cache(stepper::ExplicitRungeKuttaStepper, setup)
    @unpack NV, Np = setup.grid

    T = Float64
    ns = nstage(stepper)
    kV = zeros(T, NV, ns)
    kp = zeros(T, Np, ns)
    Vtemp = zeros(T, NV)
    Vtemp2 = zeros(T, NV)
    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)

    # Get coefficients of RK method
    A, b, c, = tableau(stepper)

    # Shift Butcher tableau, as A[1, :] is always zero for explicit methods
    A = [A[2:end, :]; b']

    # Vector with time instances (1 is the time level of final step)
    c = [c[2:end]; 1]

    ExplicitRungeKuttaStepperCache{T}(kV, kp, Vtemp, Vtemp2, F, ∇F, f, T.(A), T.(b), T.(c))
end

time_stepper_cache(::ImplicitRungeKuttaStepper) = ImplicitRungeKuttaStepperCache()
time_stepper_cache(::AdamsBashforthCrankNicolsonStepper) =
    AdamsBashforthCrankNicolsonStepperCache()
time_stepper_cache(::OneLegStepper) = OneLegStepperCache()
