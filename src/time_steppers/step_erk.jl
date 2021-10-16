"""
    step!(erk_stepper::ExplicitRungeKuttaStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ, setup, stepper_cache, momentum_cache)

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong manner via the `ubc` and `vbc` functions.
"""
function step!(::ExplicitRungeKuttaStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ, setup, stepper_cache, momentum_cache)
    @unpack Nu, Nv, Np, Ω⁻¹ = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack pressure_solver = setup.solver_settings
    @unpack time_stepper = setup.time
    @unpack kV, kp, Vtemp, Vtemp2, F, ∇F, Δp, f, A, b, c = stepper_cache

    # Number of stages
    nstage = length(b)

    # Reset RK arrays
    kV .= 0
    kp .= 0

    # Store variables at start of time step
    V .= Vₙ
    p .= pₙ

    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ)
    end

    tᵢ = tₙ

    ## Start looping over stages

    # At i = 1 we calculate F₁, p₂ and u₂
    # ⋮
    # At i = s we calculate Fₛ, pₙ₊₁, and uₙ₊₁
    for i = 1:nstage
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at, level i
        # This includes force evaluation at tᵢ and pressure gradient
        # Boundary conditions will be set through set_bc_vectors! inside momentum
        # The pressure p is not important here, it will be removed again in the
        # Next step
        momentum!(F, ∇F, V, V, p, tᵢ, setup, momentum_cache)

        # Store right-hand side of stage i
        # By adding G*p we effectively REMOVE the pressure contribution Gx*p and Gy*p (but not the vectors y_px and y_py)
        kVi = @view kV[:, i]
        mul!(kVi, G, p)
        @. kVi = Ω⁻¹ * (F + kVi)
        # kVi = Ω⁻¹ .* (F + G * p)

        # Update velocity current stage by sum of Fᵢ's until this stage,
        # Weighted with Butcher tableau coefficients
        # This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        mul!(Vtemp, kV, A[i, :])

        # Boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δtₙ
        if setup.bc.bc_unsteady
            set_bc_vectors!(setup, tᵢ)
        end

        # Divergence of intermediate velocity field
        @. Vtemp2 = Vₙ / Δtₙ + Vtemp
        mul!(f, M, Vtemp2)
        @. f = (f + yM / Δtₙ) / c[i]
        # F = (M * (Vₙ / Δtₙ + Vtemp) + yM / Δtₙ) / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if setup.bc.bc_unsteady || i > 1
            # The time tᵢ below is only for output writing
            pressure_poisson!(pressure_solver, Δp, f, tᵢ, setup)
        else
            # Bc steady AND i = 1
            Δp .= pₙ
        end

        # Store pressure
        kp[:, i] .= Δp

        mul!(Vtemp2, G, Δp)

        # Update velocity current stage, which is now divergence free
        @. V = Vₙ + Δtₙ * (Vtemp - c[i] * Ω⁻¹ * Vtemp2)
    end

    if setup.bc.bc_unsteady
        if setup.solversettings.p_add_solve
            pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F)
        else
            # Standard method
            p .= kp[:, end]
        end
    else
        # For steady bc we do an additional pressure solve
        # That saves a pressure solve for i = 1 in the next time step
        pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F)
    end

    V, p
end
