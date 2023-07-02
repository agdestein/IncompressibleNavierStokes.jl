create_stepper(::ExplicitRungeKuttaMethod; setup, pressure_solver, bc_vectors, V, p, t, n = 0) = 
    (; setup, pressure_solver, bc_vectors, V, p, t, n)

function step(method::ExplicitRungeKuttaMethod, stepper, Δt)
    (; setup, pressure_solver, bc_vectors, V, p, t, n) = stepper
    (; grid, operators, boundary_conditions) = setup
    (; bc_unsteady) = boundary_conditions
    (; Ω) = grid
    (; G, M) = operators
    (; A, b, c, p_add_solve) = method

    T = typeof(Δt)

    # Update current solution (does not depend on previous step size)
    n += 1
    Vₙ = V
    pₙ = p
    tₙ = t
    Δtₙ = Δt

    # Number of stages
    nV = length(V)
    np = length(p)
    nstage = length(b)

    # Reset RK arrays
    tᵢ = tₙ
    kV = zeros(T, nV, 0)
    kp = zeros(T, np, 0)

    ## Start looping over stages

    # At i = 1 we calculate F₁, p₂ and u₂
    # ⋮
    # At i = s we calculate Fₛ, pₙ₊₁, and uₙ₊₁
    for i = 1:nstage
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at level i. This
        # includes force evaluation at tᵢ and pressure gradient. Boundary conditions will be
        # set through `get_bc_vectors` inside momentum. The pressure p is not important here,
        # it will be removed again in the next step
        F, ∇F = momentum(V, V, p, tᵢ, setup; bc_vectors)

        # Store right-hand side of stage i
        # Remove the -G*p contribution (but not y_p)
        kVᵢ = 1 ./ Ω .* (F + G * p)
        kV = [kV kVᵢ]

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        V = kV * A[i, 1:i]

        # Boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δtₙ
        if bc_unsteady
            bc_vectors = get_bc_vectors(setup, tᵢ)
        end
        (; yM) = bc_vectors

        # Divergence of intermediate velocity field
        f = (M * (Vₙ / Δtₙ + V) + yM / Δtₙ) / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if boundary_conditions.bc_unsteady || i > 1
            p = pressure_poisson(pressure_solver, f)
        else
            # Bc steady AND i = 1
            p = pₙ
        end

        Gp = G * p

        # Update velocity current stage, which is now divergence free
        V = @. Vₙ + Δtₙ * (V - c[i] / Ω * Gp)
    end

    # For steady bc we do an additional pressure solve
    # That saves a pressure solve for i = 1 in the next time step
    if !bc_unsteady || p_add_solve
        p = pressure_additional_solve(pressure_solver, V, p, tₙ + Δtₙ, setup; bc_vectors)
    end

    t = tₙ + Δtₙ

    create_stepper(method; setup, pressure_solver, bc_vectors, V, p, t, n)
end

function step!(
    method::ExplicitRungeKuttaMethod,
    stepper,
    Δt;
    cache,
    momentum_cache,
)
    (; setup, pressure_solver, bc_vectors, V, p, t, n) = stepper
    (; grid, operators, boundary_conditions) = setup
    (; bc_unsteady) = boundary_conditions
    (; Ω) = grid
    (; G, M) = operators
    (; A, b, c, p_add_solve) = method
    (; Vₙ, pₙ, kV, kp, Vtemp, Vtemp2, F, ∇F, Δp, f) = cache

    # Update current solution (does not depend on previous step size)
    n += 1
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
        # set through `get_bc_vectors` inside momentum. The pressure p is not important here,
        # it will be removed again in the next step
        momentum!(F, ∇F, V, V, p, tᵢ, setup, momentum_cache; bc_vectors)

        # Store right-hand side of stage i
        # Remove the -G*p contribution (but not y_p)
        kVᵢ = @view kV[:, i]
        mul!(kVᵢ, G, p)
        @. kVᵢ = 1 ./ Ω * (F + kVᵢ)
        # kVᵢ .= 1 ./ Ω .* (F + G * p)

        # Update velocity current stage by sum of Fᵢ's until this stage, weighted
        # with Butcher tableau coefficients. This gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        mul!(Vtemp, kV, A[i, :])

        # Boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δtₙ
        if bc_unsteady
            bc_vectors = get_bc_vectors(setup, tᵢ)
        end
        (; yM) = bc_vectors

        # Divergence of intermediate velocity field
        @. Vtemp2 = Vₙ / Δtₙ + Vtemp
        mul!(f, M, Vtemp2)
        @. f = (f + yM / Δtₙ) / c[i]
        # f = (M * (Vₙ / Δtₙ + Vtemp) + yM / Δtₙ) / c[i]

        # Solve the Poisson equation, but not for the first step if the boundary conditions are steady
        if boundary_conditions.bc_unsteady || i > 1
            pressure_poisson!(pressure_solver, p, f)
        else
            # Bc steady AND i = 1
            p .= pₙ
        end

        mul!(Vtemp2, G, p)

        # Update velocity current stage, which is now divergence free
        @. V = Vₙ + Δtₙ * (Vtemp - c[i] / Ω * Vtemp2)
    end

    # For steady bc we do an additional pressure solve
    # That saves a pressure solve for i = 1 in the next time step
    if !bc_unsteady || p_add_solve
        pressure_additional_solve!(
            pressure_solver,
            V,
            p,
            tₙ + Δtₙ,
            setup,
            momentum_cache,
            F,
            f,
            Δp;
            bc_vectors,
        )
    end

    t = tₙ + Δtₙ

    create_stepper(method; setup, pressure_solver, bc_vectors, V, p, t, n)
end
