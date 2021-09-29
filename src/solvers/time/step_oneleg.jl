"""
One-leg β method following
Symmetry-preserving discretization of turbulent flow, Verstappen and Veldman (JCP 2003)
or:Direct numerical simulation of turbulence at lower costs (Journal of
Engineering Mathematics 1997)

formulation:
((β+1/2)*u^{n+1} -2*β*u^{n} + (β-1/2)*u^{n-1})/Δt =
 F((1+β)*u^n - β*u^{n-1})
"""
function step_oneleg!(V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δt, setup)
    @unpack G, M, yM = setup.discretization

    Om_inv = setup.grid.Om_inv
    β = setup.time.β

    # Intermediate ("offstep") velocities (see paper: "DNS at lower cost")
    t_int = tₙ + β * Δt
    V_int = (1 + β) * Vₙ - β * Vₙ₋₁
    p_int = (1 + β) * pₙ - β * pₙ₋₁

    # Right-hand side of the momentum equation
    _, F_rhs = momentum(V_int, V_int, p_int, t_int, setup)

    # Take a time step with this right-hand side, this gives an
    # intermediate velocity field (not divergence free)
    Vtemp = (2 * β * Vₙ - (β - 0.5) * Vₙ₋₁ + Δt * Om_inv .* F_rhs) / (β + 0.5)

    # To make the velocity field uₙ₊₁ at tₙ₊₁ divergence-free we need
    # the boundary conditions at tₙ₊₁
    if setup.BC.BC_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Define an adapted time step; this is only influencing the pressure calculation
    Δtᵦ = Δt / (β + 0.5)

    # Divergence of intermediate velocity field is directly calculated with M
    f = (M * Vtemp + yM) / Δtᵦ

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(f, tₙ + Δt, setup)

    # Update velocity field
    V .= Vtemp .- Δtᵦ .* Om_inv .* G * Δp

    # Update pressure (second order)
    @. p = 2pₙ - pₙ₋₁ + 4 / 3 * Δp

    # Alternatively, do an additional Poisson solve:
    if setup.solversettings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δt, setup)
    end

    V, p
end
