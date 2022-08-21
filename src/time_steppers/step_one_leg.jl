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
function step!(stepper::OneLegStepper, Δt)
    (; method, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, pressure_solver, cache, momentum_cache) = stepper
    (; p_add_solve, β) = method
    (; grid, operators, bc) = setup
    (; G, M) = operators
    (; Ω⁻¹) = grid
    (; Vₙ₋₁, pₙ₋₁, F, f, Δp, GΔp) = cache

    Δt ≈ Δtₙ || error("One-leg-β-method requires constant time step")

    # Update current solution (does not depend on previous step size)
    stepper.n += 1
    Vₙ₋₁ .= Vₙ
    pₙ₋₁ .= pₙ
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt

    # Intermediate ("offstep") velocities
    t = tₙ + β * Δtₙ
    @. V = (1 + β) * Vₙ - β * Vₙ₋₁
    @. p = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    # Take a time step with this right-hand side, this gives an intermediate velocity field
    # (not divergence free)
    @. V = (2β * Vₙ - (β - 1//2) * Vₙ₋₁ + Δtₙ * Ω⁻¹ * F) / (β + 1//2)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary
    # conditions at tₙ₊₁
    bc.bc_unsteady && set_bc_vectors!(setup, tₙ + Δtₙ)
    (; yM) = operators

    # Adapt time step for pressure calculation
    Δtᵦ = Δtₙ / (β + 1//2)

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
        pressure_additional_solve!(pressure_solver, V, p, tₙ + Δtₙ, setup, momentum_cache, F, f, Δp)
    end

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ

    stepper
end
