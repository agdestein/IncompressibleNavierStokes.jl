"""
Perform one time step with Adams-Bashforth for convection and Crank-Nicolson for diffusion.

`convₙ₋₁` are the convection terms of `tₙ₋₁`. Output includes convection terms at `tₙ`, which will be used in next time step in
the Adams-Bashforth part of the method

Adams-Bashforth for convection and Crank-Nicolson for diffusion
formulation:

(u^{n+1} - u^{n})/Δt = -(α₁ c^n + α₂ c^{n-1})
                       + θ diff^{n+1} + (1-θ) diff^{n}
                       + θ F^{n+1} + (1-θ) F^{n}
                       + θ BC^{n+1} + (1-θ) BC^{n}
                       - G*p + y_p

where BC are boundary conditions of diffusion. This is rewritten as:

(I/Δt - θ D) u^{n+1} = (I/Δt - (1-θ) D) u^{n}
                     - (α₁ c^n + α₂ c^{n-1})
                     + θ F^{n+1} + (1-θ) F^{n}
                     + θ BC^{n+1} + (1-θ) BC^{n}
                     - G*p + y_p

The LU decomposition of the first matrix is precomputed in `operator_convection_diffusion.jl`.

note that, in constrast to explicit methods, the pressure from previous
time steps has an influence on the accuracy of the velocity
"""
function step_AB_CN!(V, p, Vₙ, pₙ, convₙ₋₁, tₙ, Δt, setup, cache)
    # Adams-Bashforth coefficients
    α₁ = 3 // 2
    α₂ = -1 // 2

    @unpack Nu, Nv, indu, indv = setup.grid.Nv
    @unpack Ωu⁻¹, Ωv⁻¹, Ω⁻¹ = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Gx, Gy, y_px, y_py = setup.discretization
    @unpack yDiffu, yDiffv = setup.discretization
    @unpack Diffu, Diffv = setup.discretization
    @unpack lu_diffu, lu_diffv = setup.discretization
    @unpack θ = setup.time

    @unpack c, ∇c = cache

    uₕ = @view Vₙ[indu]
    vₕ = @view Vₙ[indv]

    # Convection from previous time step
    cuₙ₋₁ = @view convₙ₋₁[indu]
    cvₙ₋₁ = @view convₙ₋₁[indv]

    yDiffuₙ = copy(yDiffu)
    yDiffvₙ = copy(yDiffv)
    yDiffuₙ₊₁ = copy(yDiffu)
    yDiffvₙ₊₁ = copy(yDiffv)

    # Evaluate boundary conditions and force at starting point
    Fxₙ, Fyₙ, = bodyforce(Vₙ, tₙ, setup)

    # Unsteady BC at current time
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ)
    end

    # Convection of current solution
    convection!(c, ∇c, Vₙ, Vₙ, tₙ, setup, cache)

    cuₙ = @view c[indu]
    cvₙ = @view c[indv]

    # Evaluate BC and force at end of time step

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    Fxₙ₊₁, Fyₙ₊₁, = bodyforce(Vₙ, tₙ + Δt, setup)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    Fx = @. (1 - θ) * Fxₙ + θ * Fxₙ₊₁
    Fy = @. (1 - θ) * Fyₙ + θ * Fyₙ₊₁
    yDiffu = @. (1 - θ) * yDiffuₙ + θ * yDiffuₙ₊₁
    yDiffv = @. (1 - θ) * yDiffvₙ + θ * yDiffvₙ₊₁

    gxpₙ = Gx * pₙ
    gypₙ = Gy * pₙ

    # Right hand side of the momentum equation update
    Rur =
        uₕ +
        Ωu⁻¹ * Δt * (
            -(α₁ * cuₙ + α₂ * cuₙ₋₁) + (1 - θ) * Diffu * uₕ + yDiffu + Fx - Gxpₙ -
            y_px
        )
    Rvr =
        vₕ +
        Ωv⁻¹ * Δt .* (
            -(α₁ * cvₙ + α₂ * cvₙ₋₁) + (1 - θ) * Diffv * vₕ + yDiffv + Fy - Gypₙ -
            y_py
        )

    # LU decomposition of diffusion part has been calculated already in `operator_convection_diffusion.jl`
    Ru = lu_diffu \ Rur
    Rv = lu_diffv \ Rvr

    Vtemp = [Ru; Rv]

    # To make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free we need  boundary conditions at `tₙ₊₁`
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Boundary condition for the difference in pressure between time
    # steps; only non-zero in case of fluctuating outlet pressure
    y_Δp = zeros(Nu + Nv)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * Vtemp + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    Δp = pressure_poisson(f, tₙ + Δt, setup)

    # Update velocity field
    V .= Vtemp .- Δt .* Ω⁻¹ .* (G * Δp .+ y_Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if setup.solversettings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δt, setup)
    end

    V, p, c
end
