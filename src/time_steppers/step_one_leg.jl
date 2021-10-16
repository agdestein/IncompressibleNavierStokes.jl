"""
    step!(ol_stepper::OneLegStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ, setup, momentum_cache)

Do one time step using One-leg β-method following symmetry-preserving discretization of turbulent flow.
See [Verstappen and Veldman (JCP 2003)] for details,
or [Direct numerical simulation of turbulence at lower costs (Journal of Engineering Mathematics 1997)].

Formulation:
``\frac{(\beta + 1/2) u^{n+1} - 2 \beta u^{n} + (\beta - 1/2) u^{n-1}}{\Delta t} = F((1 + \beta) u^n - \beta u^{n-1})``
"""
function step!(ts::OneLegStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ, setup, stepper_cache,  momentum_cache)
    @unpack G, M, yM = setup.discretization
    @unpack pressure_solver = setup.solver_settings
    @unpack Ω⁻¹ = setup.grid
    @unpack β = ts
    @unpack F, GΔp = stepper_cache

    # Intermediate ("offstep") velocities (see paper: "DNS at lower cost")
    t = tₙ + β * Δtₙ
    @. V = (1 + β) * Vₙ - β * Vₙ₋₁
    @. p = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    # Take a time step with this right-hand side, this gives an intermediate velocity field (not divergence free)
    @. V = (2β * Vₙ - (β - 1/2) * Vₙ₋₁ + Δtₙ * Ω⁻¹ * F) / (β + 1/2)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need the boundary conditions at tₙ₊₁
    if setup.BC.BC_unsteady
        set_bc_vectors!(setup, tₙ + Δtₙ)
    end

    # Define an adapted time step; this is only influencing the pressure calculation
    Δtᵦ = Δtₙ / (β + 0.5)

    # Divergence of intermediate velocity field is directly calculated with M
    f = (M * V + yM) / Δtᵦ

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(pressure_solver, f, tₙ + Δtₙ, setup)
    mul!(GΔp, G, Δp)

    # Update velocity field
    @. V -= Δtᵦ * Ω⁻¹ * GΔp

    # Update pressure (second order)
    @. p = 2pₙ - pₙ₋₁ + 4 / 3 * Δp

    # Alternatively, do an additional Poisson solve:
    if setup.solversettings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F)
    end

    V, p
end
