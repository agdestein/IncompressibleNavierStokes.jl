"""
    step(stepper::OneLegStepper, Δt; bc_vectors = nothing)

Do one time step using one-leg-β-method.

Non-mutating/allocating/out-of-place version.

See also [`step!`](@ref).
"""
function step(stepper::OneLegStepper, Δt; bc_vectors = nothing)
    (; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ) = stepper
    (; p_add_solve, β) = method
    (; grid, operators, boundary_conditions) = setup
    (; bc_unsteady) = boundary_conditions
    (; G, M) = operators
    (; Ω⁻¹) = grid

    # Update current solution (does not depend on previous step size)
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ₋₁ = Vₙ
    pₙ₋₁ = pₙ
    Vₙ = V
    pₙ = p
    tₙ = t
    Δtₙ = Δt
    @assert Δtₙ ≈ Δtₙ₋₁

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δtₙ
    V = @. (1 + β) * Vₙ - β * Vₙ₋₁
    p = @. (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    F, = momentum(V, V, p, t, setup; bc_vectors)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    V = @. (2β * Vₙ - (β - 1 // 2) * Vₙ₋₁ + Δtₙ * Ω⁻¹ * F) / (β + 1 // 2)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δtₙ)
    end
    (; yM) = bc_vectors

    # Adapt time step for pressure calculation
    Δtᵦ = Δtₙ / (β + 1 // 2)

    # Divergence of intermediate velocity field
    f = (M * V + yM) / Δtᵦ

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(pressure_solver, f)
    GΔp = G * Δp

    # Update velocity field
    V = @. V - Δtᵦ * Ω⁻¹ * GΔp

    # Update pressure (second order)
    p = @. 2pₙ - pₙ₋₁ + 4 // 3 * Δp

    # Alternatively, do an additional Poisson solve
    if p_add_solve
        p = pressure_additional_solve(pressure_solver, V, p, tₙ + Δtₙ, setup; bc_vectors)
    end

    t = tₙ + Δtₙ

    TimeStepper(; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ)
end

"""
    step!(stepper::OneLegStepper, Δt; cache, momentum_cache, bc_vectors = nothing)

Do one time step using one-leg-β-method.
"""
function step!(stepper::OneLegStepper, Δt; cache, momentum_cache, bc_vectors = nothing)
    (; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ) = stepper
    (; p_add_solve, β) = method
    (; grid, operators, boundary_conditions) = setup
    (; bc_unsteady) = boundary_conditions
    (; G, M) = operators
    (; Ω⁻¹) = grid
    (; Vₙ₋₁, pₙ₋₁, F, f, Δp, GΔp) = cache

    # Update current solution (does not depend on previous step size)
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ₋₁ .= Vₙ
    pₙ₋₁ .= pₙ
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt
    @assert Δtₙ ≈ Δtₙ₋₁

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δtₙ
    @. V = (1 + β) * Vₙ - β * Vₙ₋₁
    @. p = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    @. V = (2β * Vₙ - (β - 1 // 2) * Vₙ₋₁ + Δtₙ * Ω⁻¹ * F) / (β + 1 // 2)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    if isnothing(bc_vectors) || bc_unsteady
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
    @. V -= Δtᵦ * Ω⁻¹ * GΔp

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

    TimeStepper(; method, setup, pressure_solver, n, V, p, t, Vₙ, pₙ, tₙ)
end
