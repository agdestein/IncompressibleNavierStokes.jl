"""
    step_ERK_ROM()

Perform one time step for the general explicit Runge-Kutta method (ERK) with Reduced Order Model (ROM).
"""
function step_erk_rom(::ExplicitRungeKuttaStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ, setup, stepper_cache, momentum_cache)
# function step_ERK_ROM(Vₙ, pₙ, tₙ, Δt, setup)
    # Number of unknowns (modes) in ROM
    M = setup.rom.M

    @unpack kV, kp, Vtemp, Vtemp2, F, ∇F, f, A, b, c = stepper_cache

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

    V, p
end
