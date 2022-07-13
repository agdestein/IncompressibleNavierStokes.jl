"""
    step!(stepper::ExplicitRungeKuttaStepper, Δt)

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong
manner via the `u_bc` and `v_bc` functions.
"""
function step!(stepper::ExplicitRungeKuttaStepper, Δt)
    (; method, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache) = stepper
    (; grid, operators, pressure_solver) = setup
    (; Ω⁻¹) = grid
    (; G, M, yM) = operators
    (; p_add_solve) = method
    (; kV, kp, Vtemp, Vtemp2, F, ∇F, Δp, f, A, b, c) = cache

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
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at level i. This
        # includes force evaluation at tᵢ and pressure gradient. Boundary conditions will be
        # set through set_bc_vectors! inside momentum. The pressure p is not important here,
        # it will be removed again in the next step
        momentum!(F, ∇F, V, V, p, tᵢ, setup, momentum_cache)

        # Store right-hand side of stage i
        # Remove the -G*p contribution (but not y_p)
        kVᵢ = @view kV[:, i]
        mul!(kVᵢ, G, p)
        @. kVᵢ = Ω⁻¹ * (F + kVᵢ)
        # kVᵢ .= Ω⁻¹ .* (F + G * p)

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        mul!(Vtemp, kV, A[i, :])

        # Boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δtₙ
        if setup.bc.bc_unsteady
            set_bc_vectors!(setup, tᵢ)
            (; yM) = setup.operators
        end

        # Divergence of intermediate velocity field
        @. Vtemp2 = Vₙ / Δtₙ + Vtemp
        mul!(f, M, Vtemp2)
        @. f = (f + yM / Δtₙ) / c[i]
        # f = (M * (Vₙ / Δtₙ + Vtemp) + yM / Δtₙ) / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if setup.bc.bc_unsteady || i > 1
            pressure_poisson!(pressure_solver, p, f)
        else
            # Bc steady AND i = 1
            p .= pₙ
        end

        # Store pressure
        kp[:, i] .= p

        mul!(Vtemp2, G, p)

        # Update velocity current stage, which is now divergence free
        @. V = Vₙ + Δtₙ * (Vtemp - c[i] * Ω⁻¹ * Vtemp2)
    end

    if setup.bc.bc_unsteady
        if p_add_solve
            pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F, f, Δp)
        else
            # Standard method
            @views p .= kp[:, end]
        end
    else
        # For steady bc we do an additional pressure solve
        # That saves a pressure solve for i = 1 in the next time step
        pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F, f, Δp)
    end

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ

    stepper
end
