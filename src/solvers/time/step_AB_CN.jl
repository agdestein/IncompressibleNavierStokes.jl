"""
# conv_old are the convection terms of t^(n-1)
# output includes convection terms at t^(n), which will be used in next time step in
# the Adams-Bashforth part of the method


## Adams-Bashforth for convection and Crank-Nicolson for diffusion
# formulation:
# (u^{n+1} - u^{n})/Δt = -(α₁*(conv^n) + α₂*(conv^{n-1})) +
#                           θ*diff^{n+1} + (1-θ)*diff^{n} +
#                           θ*F^{n+1}    + (1-θ)*F^{n}
#                           θ*bc^{n+1}   + (1-θ)*bc^{n}
#                           - G*p + y_p
# where bc are boundary conditions of diffusion

# rewrite as:
# (I/Δt - θ*D)*u^{n+1} = (I/Δt - (1-θ)*D)*u^{n} +
#                           -(α₁*(conv^n) + α₂*(conv^{n-1})) +
#                            θ*F^{n+1}    + (1-θ)*F^{n}
#                            θ*bc^{n+1} + (1-θ)*bc^{n}
#                           - G*p + y_p

# the LU decomposition of the first matrix is precomputed in
# operator_convection_diffusion

# note that, in constrast to explicit methods, the pressure from previous
# time steps has an influence on the accuracy of the velocity
"""
function step_AB_CN(Vₙ, pₙ, conv_old, tₙ, Δt, setup)

    # coefficients of the method
    # Adams-Bashforth coefficients
    α₁ = 3 / 2
    α₂ = -1 / 2

    Nu = setup.grid.Nu
    Nv = setup.grid.Nv

    Omu_inv = setup.grid.Omu_inv
    Omv_inv = setup.grid.Omv_inv
    Om_inv = setup.grid.Om_inv

    G = setup.discretization.G
    M = setup.discretization.M
    yM = setup.discretization.yM

    yDiffu2 = setup.discretization.yDiffu
    yDiffv2 = setup.discretization.yDiffv

    Gx = setup.discretization.Gx
    Gy = setup.discretization.Gy
    y_px = setup.discretization.y_px
    y_py = setup.discretization.y_py

    # diffusion of current solution
    Diffu = setup.discretization.Diffu
    Diffv = setup.discretization.Diffv
    yDiffu1 = setup.discretization.yDiffu
    yDiffv1 = setup.discretization.yDiffv

    # CN coefficients
    θ = setup.time.θ

    uₕ = Vₙ[1:Nu]
    vₕ = Vₙ[Nu+1:end]


    # convection from previous time step
    convu_old = conv_old[1:Nu]
    convv_old = conv_old[Nu+1:end]

    # evaluate bc and force at starting point
    Fx1, Fy1 = force(Vₙ, tₙ, setup, false)
    # unsteady bc at current time
    if setup.bc.bc_unsteady
        set_bc_vectors!(tₙ, setup)
    end

    # convection of current solution
    convu, convv = convection(Vₙ, Vₙ, tₙ, setup, false)

    # evaluate bc and force at end of time step

    # unsteady bc at next time
    Fx2, Fy2 = force(Vₙ, tₙ + Δt, setup, false) # Vₙ is not used normally in force.m
    if setup.bc.bc_unsteady
        set_bc_vectors!(tₙ + Δt, setup)
    end

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    Fx = (1 - θ) * Fx1 + θ * Fx2
    Fy = (1 - θ) * Fy1 + θ * Fy2
    yDiffu = (1 - θ) * yDiffu1 + θ * yDiffu2
    yDiffv = (1 - θ) * yDiffv1 + θ * yDiffv2

    # right hand side of the momentum equation update
    Rur =
        uₕ +
        Omu_inv * Δt .* (
            -(α₁ * convu + α₂ * convu_old) + (1 - θ) * Diffu * uₕ + yDiffu + Fx - Gx * pₙ -
            y_px
        )

    Rvr =
        vₕ +
        Omv_inv * Δt .* (
            -(α₁ * convv + α₂ * convv_old) + (1 - θ) * Diffv * vₕ + yDiffv + Fy - Gy * pₙ -
            y_py
        )

    # LU decomposition of diffusion part has been calculated already in
    # operator_convection_diffusion
    Ru = setup.discretization.lu_diffu \ Rur
    Rv = setup.discretization.lu_diffv \ Rvr

    Vtemp = [Ru; Rv]

    # to make the velocity field u(n+1) at t(n+1) divergence-free we need
    # the boundary conditions at t(n+1)
    if setup.bc.bc_unsteady
        set_bc_vectors!(tₙ + Δt, setup)
    end

    # boundary condition for the difference in pressure between time
    # steps; only non-zero in case of fluctuating outlet pressure
    y_Δp = zeros(Nu + Nv)

    # divergence of Ru and Rv is directly calculated with M
    f = (M * Vtemp + yM) / Δt - M * y_Δp

    # solve the Poisson equation for the pressure
    Δp = pressure_poisson(f, tₙ + Δt, setup)

    # update velocity field
    V_new = Vtemp - Δt * Om_inv .* (G * Δp + y_Δp)

    # first order pressure:
    # p_old = p;
    p_new = pₙ + Δp

    if setup.solversettings.p_add_solve
        p_new = pressure_additional_solve(V_new, pₙ, tₙ + Δt, setup)
    end

    # output convection at t^(n), to be used in next time step
    conv = [convu; convv]

    V_new, p_new, conv
end
