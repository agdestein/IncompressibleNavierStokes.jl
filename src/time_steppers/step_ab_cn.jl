"""
    step!(stepper::AdamsBashforthCrankNicolsonStepper, Δt)

Perform one time step with Adams-Bashforth for convection and Crank-Nicolson for diffusion.

`cₙ₋₁` are the convection terms of `tₙ₋₁`. Output includes convection terms at `tₙ`, which
will be used in next time step in the Adams-Bashforth part of the method.

Adams-Bashforth for convection and Crank-Nicolson for diffusion
formulation:

```math
\\begin{align*}
(\\mathbf{u}^{n+1} - \\mathbf{u}^n) / Δt & =
    -(\\alpha_1 \\mathbf{c}^n + \\alpha_2 \\mathbf{c}^{n-1}) \\\\
    & + \\theta \\mathbf{d}^{n+1} + (1-\\theta) \\mathbf{d}^n \\\\
    & + \\theta \\mathbf{F}^{n+1} + (1-\\theta) \\mathbf{F}^n \\\\
    & + \\theta \\mathbf{BC}^{n+1} + (1-\\theta) \\mathbf{BC}^n \\\\
    & - \\mathbf{G} \\mathbf{p} + \\mathbf{y}_p
\\end{align*}
```

where BC are boundary conditions of diffusion. This is rewritten as:

```math
\\begin{align*}
(\\frac{1}{\\Delta t} \\mathbf{I} - \\theta \\mathbf{D}) \\mathbf{u}^{n+1} & =
    (\\frac{1}{\\Delta t} \\mathbf{I} - (1 - \\theta) \\mathbf{D}) \\mathbf{u}^{n} \\\\
    & - (\\alpha_1 \\mathbf{c}^n + \\alpha_2 \\mathbf{c}^{n-1}) \\\\
    & + \\theta \\mathbf{F}^{n+1} + (1-\\theta) \\mathbf{F}^{n} \\\\
    & + \\theta \\mathbf{BC}^{n+1} + (1-\\theta) \\mathbf{BC}^{n} \\\\
    & - \\mathbf{G} \\mathbf{p} + \\mathbf{y}_p
\\end{align*}
```

The LU decomposition of the LHS matrix is precomputed in `operator_convection_diffusion.jl`.

Note that, in constrast to explicit methods, the pressure from previous time steps has an
influence on the accuracy of the velocity.
"""
function step!(stepper::AdamsBashforthCrankNicolsonStepper, Δt)
    (; method, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache) = stepper
    (; viscosity_model) = setup
    (; NV, Ω⁻¹) = setup.grid
    (; G, y_p, M, yM) = setup.operators
    (; Diff, yDiff, y_p) = setup.operators
    (; pressure_solver) = setup.solver_settings
    (; α₁, α₂, θ) = method
    (; cₙ, cₙ₋₁, F, f, Δp, Rr, b, bₙ, bₙ₊₁, yDiffₙ, yDiffₙ₊₁, Gpₙ, Diff_fact) = cache
    (; d, ∇d) = momentum_cache

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
        (; yDiff) = setup.operators
    end

    yDiffₙ .= yDiff

    # Evaluate boundary conditions and force at starting point
    bodyforce!(bₙ, nothing, Vₙ, tₙ, setup)

    # Convection of current solution
    convection!(cₙ, nothing, Vₙ, Vₙ, tₙ, setup, momentum_cache)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
        (; yDiff, y_p) = setup.operators
    end
    bodyforce!(bₙ₊₁, nothing, Vₙ, tₙ + Δt, setup)

    yDiffₙ₊₁ .= yDiff

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    @. b = (1 - θ) * bₙ + θ * bₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    mul!(Gpₙ, G, pₙ)
    Gpₙ .+= y_p

    mul!(d, Diff, V)

    # Right hand side of the momentum equation update
    @. Rr = Vₙ + Ω⁻¹ * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + yDiff + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        if Δt ≉ cache.Δt
            # Time step has changed, recompute LU decomposition
            Diff_fact = lu(I(NV) - θ * Δt * Diagonal(Ω⁻¹) * Diff)
            @pack! cache = Diff_fact, Δt
        end
        ldiv!(V, Diff_fact, Rr)
    else
        # Get `∇d` since `Diff` is not constant
        diffusion!(d, ∇d, V, t, setup; getJacobian = true)
        V .= ∇d \ Rr
    end

    # Make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free (need BC at `tₙ₊₁`)
    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ + Δt)
        (; yM) = setup.operators
    end

    # Boundary condition for Δp between time steps (!= 0 if fluctuating outlet pressure)
    y_Δp = zeros(NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    pressure_poisson!(pressure_solver, Δp, f)

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
