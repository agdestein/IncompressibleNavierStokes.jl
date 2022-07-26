function step(stepper::OneLegStepper, Δt)
    (; method, V, p, t, Vₙ, pₙ, tₙ, setup) = stepper
    (; p_add_solve, β) = method
    (; grid, operators, pressure_solver) = setup
    (; G, M) = operators
    (; Ω⁻¹) = grid

    Δt ≈ t - tₙ || error("One-leg-β-method requires constant time step")

    # Update current solution (does not depend on previous step size)
    Vₙ₋₁ = Vₙ
    pₙ₋₁ = pₙ
    Vₙ = V
    pₙ = p
    tₙ = t

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δt
    @. V = (1 + β) * Vₙ - β * Vₙ₋₁
    @. p = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    F = momentum(V, p, t, setup)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    V = @. (2β * Vₙ - (β - 1 // 2) * Vₙ₋₁ + Δt * Ω⁻¹ * F) / (β + 1 // 2)

    # Adapt time step for pressure calculation
    Δtᵦ = Δt / (β + 1 // 2)

    # Divergence of intermediate velocity field
    f = (M * V) / Δtᵦ

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(pressure_solver, f)
    mul!(GΔp, G, Δp)

    # Update velocity field
    V = V .- Δtᵦ .* Ω⁻¹ .* (G * Δp)

    # Update pressure (second order)
    p = @. 2pₙ - pₙ₋₁ + 4 // 3 * Δp

    # Alternatively, do an additional Poisson solve
    if p_add_solve
        # Momentum already contains G*p with the current p, we therefore
        # effectively solve for the pressure difference
        F = momentum(V, p, tₙ + Δt, setup)
        f = M * (Ω⁻¹ .* F)
        Δp = pressure_poisson(pressure_solver, f)
        p = p + Δp
    end

    n = n + 1
    t = tₙ + Δt

    OneLegStepper(; method, V, p, t, n, Vₙ, pₙ, tₙ, setup)
end

"""
    step!(stepper::OneLegStepper, Δt)

Do one time step using one-leg-β-method following symmetry-preserving discretization of
turbulent flow. See [Verstappen and Veldman (JCP 2003)] for details, or [Direct numerical
simulation of turbulence at lower costs (Journal of Engineering Mathematics 1997)].

Formulation:

```math
\\frac{(\\beta + 1/2) u^{n+1} - 2 \\beta u^{n} + (\\beta - 1/2) u^{n-1}}{\\Delta t} = F((1 +
\\beta) u^n - \\beta u^{n-1}).
```
"""
function step!(stepper::OneLegStepper, Δt; cache, momentum_cache)
    (; method, V, p, t, Vₙ, pₙ, tₙ, setup) = stepper
    (; p_add_solve, β) = method
    (; grid, operators, pressure_solver) = setup
    (; G, M) = operators
    (; Ω⁻¹) = grid
    (; Vₙ₋₁, pₙ₋₁, F, f, Δp, GΔp) = cache

    Δt ≈ t - tₙ || error("One-leg-β-method requires constant time step")

    # Update current solution (does not depend on previous step size)
    Vₙ₋₁ .= Vₙ
    pₙ₋₁ .= pₙ
    Vₙ .= V
    pₙ .= p
    tₙ = t

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δt
    @. V = (1 + β) * Vₙ - β * Vₙ₋₁
    @. p = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    momentum!(F, V, p, t, setup, momentum_cache)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    @. V = (2β * Vₙ - (β - 1 // 2) * Vₙ₋₁ + Δt * Ω⁻¹ * F) / (β + 1 // 2)

    # Adapt time step for pressure calculation
    Δtᵦ = Δt / (β + 1 // 2)

    # Divergence of intermediate velocity field
    mul!(f, M, V, 1 / Δtᵦ, 1 / Δtᵦ)
    # f .= (M * V) / Δtᵦ

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f)
    mul!(GΔp, G, Δp)

    # Update velocity field
    @. V -= Δtᵦ * Ω⁻¹ * GΔp

    # Update pressure (second order)
    @. p = 2pₙ - pₙ₋₁ + 4 // 3 * Δp

    # Alternatively, do an additional Poisson solve
    if p_add_solve
        momentum!(F, V, p, tₙ + Δt, setup, momentum_cache)
        @. F = Ω⁻¹ .* F
        mul!(f, M, F)
        pressure_poisson!(pressure_solver, Δp, f)
        p .= p .+ Δp
    end

    n = n + 1
    t = tₙ + Δt

    OneLegStepper(; method, V, p, t, n, Vₙ, pₙ, tₙ, setup)
end
