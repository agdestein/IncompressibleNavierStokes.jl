"""
    step!(stepper::ExplicitRungeKuttaStepper, Δt)

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong manner via the `u_bc` and `v_bc` functions.
"""
function step!(stepper::ExplicitRungeKuttaStepper, Δt)
    @unpack V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache = stepper
    @unpack Nu, Nv, Np, Ω⁻¹ = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack pressure_solver = setup.solver_settings
    @unpack kV, kp, Vtemp, Vtemp2, F, ∇F, Δp, f, A, b, c = cache

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
        # Remove the -G*p contribution (but not y_p)
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
            @unpack yM = setup.discretization
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
        if setup.solver_settings.p_add_solve
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

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ

    stepper
end
