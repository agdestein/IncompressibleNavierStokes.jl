"""
    step_erk_rom!(stepper::ExplicitRungeKuttaStepper, Δt)

Perform one time step for the general explicit Runge-Kutta method (ERK) with Reduced Order Model (ROM).
"""
function step_erk_rom!(stepper::ExplicitRungeKuttaStepper, Δt)
# function step_ERK_ROM(Vₙ, pₙ, tₙ, Δt, setup)
    # Number of unknowns (modes) in ROM
    @unpack M = setup.rom

    @unpack V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache = stepper 
    @unpack V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ = stepper
    @unpack kV, kp, Vtemp, Vtemp2, F, ∇F, f, A, b, c = cache

    # Update current solution (does not depend on previous step size)
    stepper.n += 1
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt

    # Number of stages
    nstage = length(b)

    # Reset RK arrays
    kV .= 0
    kp .= 0

    ## Preprocessing
    # Store variables at start of time step
    tₙ = t
    Rₙ = R

    # Right hand side evaluations, initialized at zero
    kR = zeros(M, nstage)

    # Array for the pressure
    # Kp = zeros(Np, nstage);

    tᵢ = tₙ

    for i_RK = 1:nstage
        # At i=1 we calculate F₁, p₂ and u₂
        # ...
        # At i=s we calculate Fₛ, pₙ₊₁ and uₙ₊₁

        # Right-hand side for tᵢ based on current field R at level i (this includes force evaluation at tᵢ)
        # Note that input p is not used in F_ROM
        _, F_rhs = F_ROM(R, p, tᵢ, setup)

        # Store right-hand side of stage `i`
        kR[:, i_RK] = F_rhs

        # Update coefficients R of current stage by sum of Fᵢ's until this stage,
        # Weighted with the Butcher tableau coefficients
        # This gives Rᵢ₊₁, and for `i = s` gives Rₙ₊₁
        Rtemp = kR * A[i_RK, :]

        # Time level of the computed stage
        tᵢ = tₙ + c[i_RK] * Δt

        # Update ROM coefficients current stage
        R = Rₙ + Δt * Rtemp
    end

    if setup.rom.pressure_recovery
        q = pressure_additional_solve_ROM(R, tₙ + Δt, setup)
        p = get_FOM_pressure(q, t, setup)
    end

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ   

    stepper
end
