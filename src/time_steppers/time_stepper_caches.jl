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
    ∇F::SparseMatrixCSC{T,Int}
    f::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

struct ImplicitRungeKuttaStepperCache{T} <: TimeStepperCache
    A::Any
    b::Any
    c::Any
    s::Any
    Is::Any
    Ω_sNV::Any
    A_ext::Any
    b_ext::Any
    c_ext::Any
end
struct AdamsBashforthCrankNicolsonStepperCache <: TimeStepperCache end
struct OneLegStepperCache <: TimeStepperCache end

"""
    time_stepper_cache(stepper, args...; kwargs...)

Get time stepper cache for the given time stepper.
"""
function time_stepper_cache end

function time_stepper_cache(stepper::ExplicitRungeKuttaStepper, setup)
    # TODO: Decide where `T` is to be passed
    T = Float64

    @unpack NV, Np = setup.grid

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

function time_stepper_cache(stepper::ImplicitRungeKuttaStepper)
    # TODO: Decide where `T` is to be passed
    T = Float64

    # Get coefficients of RK method
    A, b, c, = tableau(stepper)

    # Number of stages
    s = length(b)

    # Extend the Butcher tableau
    Is = sparse(I, s, s)
    Ω_sNV = kron(Is, spdiagm(Ω))
    A_ext = kron(A, sparse(I, NV, NV))
    b_ext = kron(b', sparse(I, NV, NV))
    c_ext = spdiagm(c)

    ImplicitRungeKuttaStepperCache{T}(A, b, c, s, Is, Ω_sNV, A_ext, b_ext, c_ext)
end
time_stepper_cache(::AdamsBashforthCrankNicolsonStepper) =
    AdamsBashforthCrankNicolsonStepperCache()
time_stepper_cache(::OneLegStepper) = OneLegStepperCache()
