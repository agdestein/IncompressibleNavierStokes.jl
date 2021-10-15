"""
    step!(ol_stepper::OneLegStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δt, setup, cache)

Do one time step using One-leg β-method following symmetry-preserving discretization of turbulent flow.
See [Verstappen and Veldman (JCP 2003)] for details,
or [Direct numerical simulation of turbulence at lower costs (Journal of Engineering Mathematics 1997)].

Formulation:
((β+1/2) * u^{n+1} - 2*β*u^{n} + (β-1/2)*u^{n-1}) / Δt = F((1+β) * u^n - β*u^{n-1})
"""
function step!(::OneLegStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δt, setup, cache)
    @unpack G, M, yM = setup.discretization
    @unpack pressure_solver = setup.solver_settings

    Ω⁻¹ = setup.grid.Ω⁻¹
    β = setup.time.β

    # Intermediate ("offstep") velocities (see paper: "DNS at lower cost")
    t_int = tₙ + β * Δt
    V_int = (1 + β) * Vₙ - β * Vₙ₋₁
    p_int = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    momentum!(F_rhs, nothing, V_int, V_int, p_int, t_int, setup, cache)

    # Take a time step with this right-hand side, this gives an
    # Intermediate velocity field (not divergence free)
    Vtemp = (2 * β * Vₙ - (β - 0.5) * Vₙ₋₁ + Δt * Ω⁻¹ .* F_rhs) / (β + 0.5)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need
    # The boundary conditions at tₙ₊₁
    if setup.BC.BC_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Define an adapted time step; this is only influencing the pressure calculation
    Δtᵦ = Δt / (β + 0.5)

    # Divergence of intermediate velocity field is directly calculated with M
    f = (M * Vtemp + yM) / Δtᵦ

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(pressure_solver, f, tₙ + Δt, setup)

    # Update velocity field
    V .= Vtemp .- Δtᵦ .* Ω⁻¹ .* G * Δp

    # Update pressure (second order)
    @. p = 2pₙ - pₙ₋₁ + 4 / 3 * Δp

    # Alternatively, do an additional Poisson solve:
    if setup.solversettings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δt, setup, cache, F)
    end

    V, p
end
