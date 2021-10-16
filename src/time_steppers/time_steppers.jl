"""
    TimeStepper

Abstract time stepper.
"""
abstract type TimeStepper end

"""
    AdamsBashforthCrankNicolsonStepper(; α₁ = 3 // 2, α₂ = -1 // 2, θ = 1 // 2)

Adams-Bashforth for velocity and Crank-Nicolson for pressure (with implicitness `θ`).
"""
Base.@kwdef struct AdamsBashforthCrankNicolsonStepper{T} <: TimeStepper
    α₁::T = 3 // 2
    α₂::T = -1 // 2
    θ::T = 1 // 2
end

"""
    OneLegStepper(β = 1 // 2)

One-leg β-method.
"""
Base.@kwdef struct OneLegStepper{T} <: TimeStepper
    β::T = 1 // 2
end

"""
    RungeKuttaStepper

Abstract Runge Kutta time stepper.
"""
abstract type RungeKuttaStepper <: TimeStepper end
abstract type ExplicitRungeKuttaStepper <: RungeKuttaStepper end
abstract type ImplicitRungeKuttaStepper <: RungeKuttaStepper end


## ================SSP Methods=========================

# Explicit Methods
struct FE11 <: ExplicitRungeKuttaStepper end
struct SSP22 <: ExplicitRungeKuttaStepper end
struct SSP42 <: ExplicitRungeKuttaStepper end
struct SSP33 <: ExplicitRungeKuttaStepper end
struct SSP43 <: ExplicitRungeKuttaStepper end
struct SSP104 <: ExplicitRungeKuttaStepper end
struct rSSPs2 <: ExplicitRungeKuttaStepper end
struct rSSPs3 <: ExplicitRungeKuttaStepper end
struct Wray3 <: ExplicitRungeKuttaStepper end
struct RK56 <: ExplicitRungeKuttaStepper end
struct DOPRI6 <: ExplicitRungeKuttaStepper end

# Implicit Methods
struct BE11 <: ImplicitRungeKuttaStepper end
struct SDIRK34 <: ImplicitRungeKuttaStepper end
struct ISSPm2 <: ImplicitRungeKuttaStepper end
struct ISSPs3 <: ImplicitRungeKuttaStepper end

# Half explicit methods
struct HEM3 <: ImplicitRungeKuttaStepper end
struct HEM3BS <: ImplicitRungeKuttaStepper end
struct HEM5 <: ImplicitRungeKuttaStepper end

# Classical Methods
struct GL1 <: ExplicitRungeKuttaStepper end
struct GL2 <: ExplicitRungeKuttaStepper end
struct GL3 <: ExplicitRungeKuttaStepper end
struct RIA1 <: ExplicitRungeKuttaStepper end
struct RIA2 <: ExplicitRungeKuttaStepper end
struct RIA3 <: ExplicitRungeKuttaStepper end
struct RIIA1 <: ExplicitRungeKuttaStepper end
struct RIIA2 <: ExplicitRungeKuttaStepper end
struct RIIA3 <: ExplicitRungeKuttaStepper end
struct LIIIA2 <: ExplicitRungeKuttaStepper end
struct LIIIA3 <: ExplicitRungeKuttaStepper end

# Chebyshev methods
struct CHDIRK3 <: ExplicitRungeKuttaStepper end
struct CHCONS3 <: ExplicitRungeKuttaStepper end
struct CHC3 <: ExplicitRungeKuttaStepper end
struct CHC5 <: ExplicitRungeKuttaStepper end

# Miscellaneous Methods
struct Mid22 <: ExplicitRungeKuttaStepper end
struct MTE22 <: ExplicitRungeKuttaStepper end
struct CN22 <: ExplicitRungeKuttaStepper end
struct Heun33 <: ExplicitRungeKuttaStepper end
struct RK33C2 <: ExplicitRungeKuttaStepper end
struct RK33P2 <: ExplicitRungeKuttaStepper end
struct RK44 <: ExplicitRungeKuttaStepper end
struct RK44C2 <: ExplicitRungeKuttaStepper end
struct RK44C23 <: ExplicitRungeKuttaStepper end
struct RK44P2 <: ExplicitRungeKuttaStepper end

# DSRK Methods
struct DSso2 <: ExplicitRungeKuttaStepper end
struct DSRK2 <: ExplicitRungeKuttaStepper end
struct DSRK3 <: ExplicitRungeKuttaStepper end

# "Non-SSP" Methods of Wong & Spiteri
struct NSSP21 <: ExplicitRungeKuttaStepper end
struct NSSP32 <: ExplicitRungeKuttaStepper end
struct NSSP33 <: ExplicitRungeKuttaStepper end
struct NSSP53 <: ExplicitRungeKuttaStepper end
