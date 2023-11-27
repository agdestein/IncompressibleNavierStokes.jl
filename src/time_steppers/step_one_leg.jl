create_stepper(
    method::OneLegMethod;
    setup,
    pressure_solver,
    bc_vectors,
    V,
    p,
    t,
    n = 0,

    # For the first step, these are not used
    Vₙ = copy(V),
    pₙ = copy(p),
    tₙ = t,
) = (; setup, pressure_solver, bc_vectors, V, p, t, n, Vₙ, pₙ, tₙ)

function timestep(method::OneLegMethod, stepper, Δt)
    (; setup, pressure_solver, bc_vectors, V, p, t, n, Vₙ, pₙ, tₙ) = stepper
    (; p_add_solve, β, method_startup) = method
    (; grid, operators, boundary_conditions) = setup
    (; bc_unsteady) = boundary_conditions
    (; G, M) = operators
    (; Ω) = grid

    # One-leg requires state at previous time step, which is not available at
    # the first iteration. Do one startup step instead
    if n == 0
        stepper_startup =
            create_stepper(method_startup; setup, pressure_solver, bc_vectors, V, p, t)
        n += 1
        Vₙ = V
        pₙ = p
        tₙ = t
        (; V, p, t) = timestep(method_startup, stepper_startup, Δt)
        return create_stepper(
            method;
            setup,
            pressure_solver,
            bc_vectors,
            V,
            p,
            t,
            n,
            Vₙ,
            pₙ,
            tₙ,
        )
    end

    # Update current solution
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ₋₁ = Vₙ
    pₙ₋₁ = pₙ
    Vₙ = V
    pₙ = p
    tₙ = t
    Δtₙ = Δt

    # One-leg requires fixed time step
    @assert Δtₙ ≈ Δtₙ₋₁

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δtₙ
    V = @. (1 + β) * Vₙ - β * Vₙ₋₁
    p = @. (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    F, = momentum(V, V, p, t, setup; bc_vectors)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    V = @. (2β * Vₙ - (β - 1 // 2) * Vₙ₋₁ + Δtₙ / Ω * F) / (β + 1 // 2)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    if bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δtₙ)
    end
    (; yM) = bc_vectors

    # Adapt time step for pressure calculation
    Δtᵦ = Δtₙ / (β + 1 // 2)

    # Divergence of intermediate velocity field
    f = (M * V + yM) / Δtᵦ

    # Solve the Poisson equation for the pressure
    Δp = poisson(pressure_solver, f)
    GΔp = G * Δp

    # Update velocity field
    V = @. V - Δtᵦ / Ω * GΔp

    # Update pressure (second order)
    p = @. 2pₙ - pₙ₋₁ + 4 // 3 * Δp

    # Alternatively, do an additional Poisson solve
    if p_add_solve
        p = pressure_additional_solve(pressure_solver, V, p, tₙ + Δtₙ, setup; bc_vectors)
    end

    t = tₙ + Δtₙ

    create_stepper(method; setup, pressure_solver, bc_vectors, V, p, t, n, Vₙ, pₙ, tₙ)
end

function timestep!(method::OneLegMethod, stepper, Δt; cache, momentum_cache)
    (; setup, pressure_solver, bc_vectors, n, V, p, t, Vₙ, pₙ, tₙ) = stepper
    (; p_add_solve, β, method_startup) = method
    (; grid, operators, boundary_conditions) = setup
    (; bc_unsteady) = boundary_conditions
    (; G, M) = operators
    (; Ω) = grid
    (; Vₙ₋₁, pₙ₋₁, F, f, Δp, GΔp) = cache

    # One-leg requires state at previous time step, which is not available at
    # the first iteration. Do one startup step instead
    if n == 0
        stepper_startup =
            create_stepper(method_startup; setup, pressure_solver, bc_vectors, V, p, t)
        n += 1
        Vₙ = V
        pₙ = p
        tₙ = t

        # Note: We do one out-of-place step here, with a few allocations
        (; V, p, t) = timestep(method_startup, stepper_startup, Δt)
        return create_stepper(
            method;
            setup,
            pressure_solver,
            bc_vectors,
            V,
            p,
            t,
            n,
            Vₙ,
            pₙ,
            tₙ,
        )
    end

    # Update current solution
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ₋₁ .= Vₙ
    pₙ₋₁ .= pₙ
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt

    # One-leg requires fixed time step
    @assert Δtₙ ≈ Δtₙ₋₁

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δtₙ
    @. V = (1 + β) * Vₙ - β * Vₙ₋₁
    @. p = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    @. V = (2β * Vₙ - (β - 1 // 2) * Vₙ₋₁ + Δtₙ / Ω * F) / (β + 1 // 2)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    if bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δtₙ)
    end
    (; yM) = bc_vectors

    # Adapt time step for pressure calculation
    Δtᵦ = Δtₙ / (β + 1 // 2)

    # Divergence of intermediate velocity field
    f .= yM
    mul!(f, M, V, 1 / Δtᵦ, 1 / Δtᵦ)
    # f .= (M * V + yM) / Δtᵦ

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f)
    mul!(GΔp, G, Δp)

    # Update velocity field
    @. V -= Δtᵦ / Ω * GΔp

    # Update pressure (second order)
    @. p = 2pₙ - pₙ₋₁ + 4 // 3 * Δp

    # Alternatively, do an additional Poisson solve
    if p_add_solve
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

    create_stepper(method; setup, pressure_solver, bc_vectors, V, p, t, n, Vₙ, pₙ, tₙ)
end
