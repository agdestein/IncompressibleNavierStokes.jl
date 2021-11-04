"""
    step!(stepper::AdamsBashforthCrankNicolsonStepper, Δt)

Perform one time step with Adams-Bashforth for convection and Crank-Nicolson for diffusion.

`cₙ₋₁` are the convection terms of `tₙ₋₁`. Output includes convection terms at `tₙ`, which
will be used in next time step in the Adams-Bashforth part of the method.

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

The LU decomposition of the LHS matrix is precomputed in `operator_convection_diffusion.jl`.

Note that, in constrast to explicit methods, the pressure from previous time steps has an
influence on the accuracy of the velocity.
"""
function step!(stepper::AdamsBashforthCrankNicolsonStepper, Δt)

    @unpack method, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache = stepper
    @unpack NV, Ω⁻¹ = setup.grid
    @unpack G, y_p, M, yM = setup.discretization
    @unpack Diff, yDiff, y_p = setup.discretization
    @unpack pressure_solver = setup.solver_settings
    @unpack α₁, α₂, θ = method
    @unpack cₙ, cₙ₋₁, F, Δp, Rr, b, bₙ, bₙ₊₁, yDiffₙ, yDiffₙ₊₁, Gpₙ, Diff_fact = cache
    @unpack d = momentum_cache

    Δt ≈ Δtₙ || error("Adams-Bashforth requires constant time step")

    # For the first time step, this might be necessary
    convection!(cₙ, nothing, Vₙ, Vₙ, tₙ, setup, momentum_cache)

    # Advance one step
    stepper.n += 1
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt
    cₙ₋₁ .= cₙ

    # Unsteady BC at current time
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ)
        @unpack yDiff = setup.discretization
    end

    yDiffₙ .= yDiff
    yDiffₙ₊₁ .= yDiff

    # Evaluate boundary conditions and force at starting point
    bodyforce!(bₙ, nothing, Vₙ, tₙ, setup)

    # Convection of current solution
    convection!(cₙ, nothing, Vₙ, Vₙ, tₙ, setup, momentum_cache)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
        @unpack y_p = setup.discretization
    end
    bodyforce!(bₙ₊₁, nothing, Vₙ, tₙ + Δt, setup)

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    @. b = (1 - θ) * bₙ + θ * bₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    mul!(Gpₙ, G, pₙ)
    Gpₙ .+= y_p

    mul!(d, Diff, V)

    # Right hand side of the momentum equation update
    @. Rr = Vₙ + Ω⁻¹ * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + yDiff + b - Gpₙ)

    # Use precomputed LU decomposition
    ldiv!(V, Diff_fact, Rr)

    # Make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free (need BC at `tₙ₊₁`)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
        @unpack yM = setup.discretization
    end

    # Boundary condition for Δp between time steps (!= 0 if fluctuating outlet pressure)
    y_Δp = zeros(NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f, tₙ + Δt, setup)

    # Update velocity field
    V .-= Δt .* Ω⁻¹ .* (G * Δp .+ y_Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if setup.solver_settings.p_add_solve
        pressure_additional_solve!(V, p, tₙ + Δt, setup, momentum_cache, F, f, Δp)
    end

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ

    stepper
end
