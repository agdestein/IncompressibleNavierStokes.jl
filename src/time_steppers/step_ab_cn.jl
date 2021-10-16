"""
    step!(ab_cn_stepper::AdamsBashforthCrankNicolsonStepper, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, cₙ₋₁, tₙ, Δt, setup, momentum_cache)

Perform one time step with Adams-Bashforth for convection and Crank-Nicolson for diffusion.

`cₙ₋₁` are the convection terms of `tₙ₋₁`. Output includes convection terms at `tₙ`, which will be used in next time step in
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
function step!(
    ts::AdamsBashforthCrankNicolsonStepper,
    V,
    p,
    Vₙ,
    pₙ,
    Vₙ₋₁,
    pₙ₋₁,
    cₙ₋₁,
    tₙ,
    Δt,
    setup,
    stepper_cache,
    momentum_cache,
)
    @unpack NV, Ω⁻¹ = setup.grid
    @unpack G, y_p, M, yM = setup.discretization
    @unpack Diff, yDiff = setup.discretization
    @unpack pressure_solver = setup.solver_settings
    @unpack α₁, α₂, θ = ts
    @unpack F, Δp, Diff_fact = stepper_cache
    @unpack c, ∇c, d, ∇d = momentum_cache

    yDiffₙ = copy(yDiff)
    yDiffₙ₊₁ = copy(yDiff)

    # Evaluate boundary conditions and force at starting point
    bodyforce!(F, nothing, Vₙ, tₙ, setup)

    # Unsteady BC at current time
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ)
    end

    # Convection of current solution
    convection!(c, ∇c, Vₙ, Vₙ, tₙ, setup, momentum_cache)

    # Evaluate BC and force at end of time step

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    bodyforce!(F, nothing, Vₙ, tₙ + Δt, setup)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    F = @. (1 - θ) * Fₙ + θ * Fₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    Gpₙ = G * pₙ

    mul!(d, Diff, V)
    d .+ yDiff

    # Right hand side of the momentum equation update
    Rr = @. Vₙ + Ω⁻¹ * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + F - Gpₙ - y_p)

    # LU decomposition of diffusion part has been calculated already in `operator_convection_diffusion.jl`
    Vtemp = Diff_fact \ Rr

    # To make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free we need  boundary conditions at `tₙ₊₁`
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
    end

    # Boundary condition for the difference in pressure between time
    # steps; only non-zero in case of fluctuating outlet pressure
    y_Δp = zeros(NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * Vtemp + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f, tₙ + Δt, setup)

    # Update velocity field
    V .= Vtemp .- Δt .* Ω⁻¹ .* (G * Δp .+ y_Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if setup.solver_settings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δt, setup)
    end

    c
end
