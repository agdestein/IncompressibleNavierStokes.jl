"""
one-leg β method following
Symmetry-preserving discretization of turbulent flow, Verstappen and Veldman (JCP 2003)
or:Direct numerical simulation of turbulence at lower costs (Journal of
Engineering Mathematics 1997)

formulation:
((β+1/2)*u^{n+1} -2*β*u^{n} + (β-1/2)*u^{n-1})/Δt =
 F((1+β)*u^n - β*u^{n-1})
"""
function step_oneleg(Vₙ, pₙ, V_old, p_old, tₙ, Δt, setup)
    @unpack G, M, yM = setup.discretization

    Om_inv = setup.grid.Om_inv
    β = setup.time.β

    ## preprocessing

    ## take time step

    # intermediate ("offstep") velocities
    t_int = tₙ + β * Δt
    V_int = (1 + β) * Vₙ - β * V_old
    p_int = (1 + β) * pₙ - β * p_old # see paper: "DNS at lower cost"
    #p_temp = p;

    # right-hand side of the momentum equation
    _, F_rhs = F(V_int, V_int, p_int, t_int, setup)

    # take a time step with this right-hand side, this gives an
    # intermediate velocity field (not divergence free)
    Vtemp = (2 * β * Vₙ - (β - 0.5) * V_old + Δt * Om_inv .* F_rhs) / (β + 0.5)

    # to make the velocity field u(n+1) at t(n+1) divergence-free we need
    # the boundary conditions at t(n+1)
    if setup.BC.BC_unsteady
        set_bc_vectors!(tₙ + Δt, setup)
    end


    # define an adapted time step; this is only influencing the pressure
    # calculation
    Δtᵦ = Δt / (β + 0.5)

    # divergence of intermediate velocity field is directly calculated with M
    f = (M * Vtemp + yM) / Δtᵦ

    # solve the Poisson equation for the pressure
    Δp = pressure_poisson(f, tₙ + Δt, setup)

    # update velocity field
    V_new = Vtemp - Δtᵦ * Om_inv .* G * Δp

    # update pressure (second order)
    p_new = 2pₙ - p_old + 4 / 3 * Δp

    # alternatively, do an additional Poisson solve:
    if setup.solversettings.p_add_solve
        p_new = pressure_additional_solve(V_new, pₙ, tₙ + Δt, setup)
    end

    V_new, p_new
end
