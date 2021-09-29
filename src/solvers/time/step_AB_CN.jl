"""
convₙ₋₁ are the convection terms of t^(n-1)
output includes convection terms at t^(n), which will be used in next time step in
the Adams-Bashforth part of the method

Adams-Bashforth for convection and Crank-Nicolson for diffusion
formulation:
(u^{n+1} - u^{n})/Δt = -(α₁*(conv^n) + α₂*(conv^{n-1})) +
                          θ*diff^{n+1} + (1-θ)*diff^{n} +
                          θ*F^{n+1}    + (1-θ)*F^{n}
                          θ*BC^{n+1}   + (1-θ)*BC^{n}
                          - G*p + y_p
where BC are boundary conditions of diffusion

rewrite as:
(I/Δt - θ*D)*u^{n+1} = (I/Δt - (1-θ)*D)*u^{n} +
                          -(α₁*(conv^n) + α₂*(conv^{n-1})) +
                           θ*F^{n+1}    + (1-θ)*F^{n}
                           θ*BC^{n+1} + (1-θ)*BC^{n}
                          - G*p + y_p

the LU decomposition of the first matrix is precomputed in
operator_convection_diffusion

note that, in constrast to explicit methods, the pressure from previous
time steps has an influence on the accuracy of the velocity
"""
function step_AB_CN!(V, p, Vₙ, pₙ, convₙ₋₁, tₙ, Δt, setup)
    # Adams-Bashforth coefficients
    α₁ = 3 // 2
    α₂ = -1 // 2

    @unpack Nu, Nv = setup.grid.Nv
    @unpack Omu_inv, Omv_inv, Om_inv = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Gx, Gy, y_px, y_py = setup.discretization
    @unpack yDiffu, yDiffv = setup.discretization
    @unpack Diffu, Diffv = setup.discretization
    @unpack lu_diffu, lu_diffv = setup.discretization
    @unpack θ = setup.time

    uₕ = @view Vₙ[1:Nu]
    vₕ = @view Vₙ[Nu+1:end]

    # convection from previous time step
    convuₙ₋₁ = @view convₙ₋₁[1:Nu]
    convvₙ₋₁ = @view convₙ₋₁[Nu+1:end]

    yDiffuₙ = yDiffu
    yDiffvₙ = yDiffv
    yDiffuₙ₊₁ = yDiffu
    yDiffvₙ₊₁ = yDiffv

    # evaluate bc and force at starting point
    Fxₙ, Fyₙ = force(Vₙ, tₙ, setup, false)

    # unsteady bc at current time
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ)
    end

    # Convection of current solution
    convuₙ, convvₙ = convection(Vₙ, Vₙ, tₙ, setup, false)

    # Evaluate BC and force at end of time step

    # Unsteady BC at next time
    Fxₙ₊₁, Fyₙ₊₁ = force(Vₙ, tₙ + Δt, setup, false) # Vₙ is not used normally in force.jl
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    Fx = (1 - θ) * Fxₙ + θ * Fxₙ₊₁
    Fy = (1 - θ) * Fyₙ + θ * Fyₙ₊₁
    yDiffu = (1 - θ) * yDiffuₙ + θ * yDiffuₙ₊₁
    yDiffv = (1 - θ) * yDiffvₙ + θ * yDiffvₙ₊₁

    # right hand side of the momentum equation update
    Rur =
        uₕ +
        Omu_inv * Δt .* (
            -(α₁ * convuₙ + α₂ * convuₙ₋₁) + (1 - θ) * Diffu * uₕ + yDiffu + Fx - Gx * pₙ -
            y_px
        )
    Rvr =
        vₕ +
        Omv_inv * Δt .* (
            -(α₁ * convvₙ + α₂ * convvₙ₋₁) + (1 - θ) * Diffv * vₕ + yDiffv + Fy - Gy * pₙ -
            y_py
        )

    # LU decomposition of diffusion part has been calculated already in
    # operator_convection_diffusion
    Ru = lu_diffu \ Rur
    Rv = lu_diffv \ Rvr

    Vtemp = [Ru; Rv]

    # To make the velocity field u(n+1) at t(n+1) divergence-free we need
    # the boundary conditions at t(n+1)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Boundary condition for the difference in pressure between time
    # steps; only non-zero in case of fluctuating outlet pressure
    y_Δp = zeros(Nu + Nv)

    # Divergence of Ru and Rv is directly calculated with M
    f = (M * Vtemp + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(f, tₙ + Δt, setup)

    # Update velocity field
    V .= Vtemp .- Δt .* Om_inv .* (G * Δp .+ y_Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if setup.solversettings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δt, setup)
    end

    # output convection at tₙ, to be used in next time step
    conv = [convuₙ; convvₙ]

    V, p, conv
end
