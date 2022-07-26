"""
    step(stepper::ExplicitRungeKuttaStepper, Δt)

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong
manner via the `u_bc` and `v_bc` functions.
"""
function step(stepper::ExplicitRungeKuttaStepper, Δt)
    (; method, V, p, t, n, setup) = stepper
    (; grid, operators, pressure_solver) = setup
    (; Ω⁻¹) = grid
    (; G, M) = operators
    (; A, b, c, p_add_solve) = method

    # Update current solution (does not depend on previous step size)
    Vₙ = V
    pₙ = p
    tₙ = t

    # Number of stages
    nV = length(V)
    nstage = length(b)

    # Reset RK arrays
    tᵢ = tₙ
    kV = zeros(nV, 0)

    ## Start looping over stages

    # At i = 1 we calculate F₁, p₂ and u₂
    # ⋮
    # At i = s we calculate Fₛ, pₙ₊₁, and uₙ₊₁
    for i = 1:nstage
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at level i. This
        # includes force evaluation at tᵢ and pressure gradient. Boundary conditions will be
        # set through set_bc_vectors! inside momentum. The pressure p is not important here,
        # it will be removed again in the next step
        F = momentum(V, p, tᵢ, setup)

        # Store right-hand side of stage i
        # Remove the -G*p contribution
        kVᵢ = Ω⁻¹ .* (F .+ G * p)
        kV = [kV kVᵢ]

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        tᵢ = tₙ + c[i] * Δt
        V = kV * A[i, 1:i]

        # Divergence of intermediate velocity field
        f = (M * (Vₙ / Δt + V)) / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if i == 1
            p = pₙ
        else
            p = pressure_poisson(pressure_solver, f)
        end

        # Update velocity current stage, which is now divergence free
        V = Vₙ + Δt * (V - c[i] * Ω⁻¹ .* (G * p))
    end

    # That saves a pressure solve for i = 1 in the next time step
    if p_add_solve
        # Momentum already contains G*p with the current p, we therefore
        # effectively solve for the pressure difference
        F = momentum(V, p, tₙ + Δt, setup)
        f = M * (Ω⁻¹ .* F)
        Δp = pressure_poisson(pressure_solver, f)
        p + Δp
    end

    t = tₙ + Δt
    n = n + 1

    ExplicitRungeKuttaStepper(; method, V, p, t, n, setup)
end

"""
    step!(stepper::ExplicitRungeKuttaStepper, Δt)

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong
manner via the `u_bc` and `v_bc` functions.
"""
function step!(stepper::ExplicitRungeKuttaStepper, Δt; cache, momentum_cache)
    (; method, V, p, t, n, setup) = stepper
    (; grid, operators, pressure_solver) = setup
    (; Ω⁻¹) = grid
    (; G, M) = operators
    (; Vₙ, pₙ, kV, Vtemp, F, ∇F, Δp, f) = cache
    (; A, b, c, p_add_solve) = method

    # Update current solution (does not depend on previous step size)
    Vₙ .= V
    pₙ .= p
    tₙ = t

    # Number of stages
    nstage = length(b)

    # Reset RK arrays
    tᵢ = tₙ
    kV .= 0

    ## Start looping over stages

    # At i = 1 we calculate F₁, p₂ and u₂
    # ⋮
    # At i = s we calculate Fₛ, pₙ₊₁, and uₙ₊₁
    for i = 1:nstage
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at level i. This
        # includes force evaluation at tᵢ and pressure gradient. Boundary conditions will be
        # set through set_bc_vectors! inside momentum. The pressure p is not important here,
        # it will be removed again in the next step
        momentum!(F, V, p, tᵢ, setup, momentum_cache)

        # Store right-hand side of stage i
        # Remove the -G*p contribution
        kVᵢ = @view kV[:, i]
        mul!(kVᵢ, G, p)
        @. kVᵢ = Ω⁻¹ * (F + kVᵢ)

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        mul!(V, kV, A[i, :])

        # Boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δt

        # Divergence of intermediate velocity field
        @. Vtemp = Vₙ / Δt + V
        mul!(f, M, Vtemp)
        @. f = f / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if i == 1
            p .= pₙ
        else
            pressure_poisson!(pressure_solver, p, f)
        end

        mul!(Vtemp, G, p)

        # Update velocity current stage, which is now divergence free
        @. V = Vₙ + Δt * (V - c[i] * Ω⁻¹ * Vtemp)
    end

    # That saves a pressure solve for i = 1 in the next time step
    if p_add_solve
        # Momentum already contains G*p with the current p, we therefore
        # effectively solve for the pressure difference
        momentum!(F, V, p, tₙ + Δt, setup, momentum_cache)
        @. F = Ω⁻¹ .* F
        mul!(f, M, F)
        pressure_poisson!(pressure_solver, Δp, f)
        p .= p .+ Δp
    end

    t = tₙ + Δt
    n = n + 1

    ExplicitRungeKuttaStepper(; method, V, p, t, n, setup)
end
