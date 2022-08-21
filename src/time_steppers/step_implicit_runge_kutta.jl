"""
    step!(stepper::ImplicitRungeKuttaStepper, Δt)

Do one time step for implicit Runge-Kutta method.

Unsteady Dirichlet boundary points are not part of solution vector but
are prescribed in a "strong" manner via the `u_bc` and `v_bc` functions.
"""
function step!(stepper::ImplicitRungeKuttaStepper, Δt)
    (; method, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, pressure_solver, cache, momentum_cache) = stepper
    (; grid, operators) = setup
    (; NV, Np, Ω⁻¹) = grid
    (; G, M) = operators
    (; p_add_solve, maxiter, abstol, newton_type) = method
    (; Vtotₙ, ptotₙ, Qⱼ, Fⱼ, ∇Fⱼ, fⱼ, F, ∇F, f, Δp, Gp) = cache
    (; Mtot, yMtot, Ωtot, dfmom, Z) = cache
    (; A, b, c, Ω_sNV, A_ext, b_ext) = cache

    is_patterned = stepper.n > 1

    # Update current solution (does not depend on previous step size)
    stepper.n += 1
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt

    # Number of stages
    s = length(b)

    # Time instances at all stages, tⱼ = [t₁, t₂, ..., tₛ]
    tⱼ = @. tₙ + c * Δtₙ

    Vtotₙ_mat = reshape(Vtotₙ, :, s)
    ptotₙ_mat = reshape(ptotₙ, :, s)
    yMtot_mat = reshape(yMtot, :, s)
    for i = 1:s
        # Initialize with the solution at tₙ
        Vtotₙ_mat[:, i] .= Vₙ
        ptotₙ_mat[:, i] .= pₙ

        # Boundary conditions at all the stage time steps to make the velocity field
        # uᵢ₊₁ at tᵢ₊₁ divergence-free (BC at tᵢ₊₁ needed)
        if i == 1 || setup.bc.bc_unsteady
            # Modify `yM`
            tᵢ = tⱼ[i]
            set_bc_vectors!(setup, tᵢ)
            (; yM) = setup.operators
        end
        yMtot_mat[:, i] .= yM
    end

    # Iteration counter
    iter = 0

    # Index in global solution vector
    ind_Vⱼ = 1:(NV * s)
    ind_pⱼ = (NV * s + 1):((NV + Np) * s)

    # Vtot contains all stages and is ordered as [u₁; v₁; u₂; v₂; ...; uₛ; vₛ];
    # Starting guess for intermediate stages
    # This can be improved, see e.g. the Radau, Gauss4, or Lobatto scripts
    Vⱼ = @view Qⱼ[ind_Vⱼ]
    pⱼ = @view Qⱼ[ind_pⱼ]
    Vⱼ .= Vtotₙ
    pⱼ .= ptotₙ

    # Initialize right-hand side for all stages
    momentum_allstage!(Fⱼ, ∇Fⱼ, Vⱼ, Vⱼ, pⱼ, tⱼ, setup, cache, momentum_cache)

    fmomⱼ = @view fⱼ[ind_Vⱼ]
    fmassⱼ = @view fⱼ[ind_pⱼ]

    # Initialize momentum residual
    mul!(fmomⱼ, A_ext, Fⱼ)
    @. fmomⱼ -= (Ωtot * Vⱼ - Ωtot * Vtotₙ) / Δtₙ
    # fmomⱼ .= .-(Ωtot .* Vⱼ .- Ωtot .* Vtotₙ) ./ Δtₙ .+ A_ext * Fⱼ

    # Initialize mass residual
    fmassⱼ .= yMtot
    mul!(fmassⱼ, Mtot, Vⱼ, -1, -1)
    # fmassⱼ .= .-(Mtot * Vⱼ .+ yMtot)

    if newton_type == :approximate
        # Approximate Newton (Jacobian is based on current solution Vₙ)
        momentum!(F, ∇F, Vₙ, Vₙ, pₙ, tₙ, setup, momentum_cache; getJacobian = true)

        # Update iteration matrix, which is now fixed during iterations
        if is_patterned
            kron!(dfmom, A, ∇F)
        else
            dfmom .= kron(A, ∇F)
        end
        @. dfmom = Ω_sNV / Δtₙ - dfmom
        # dfmom .= Ω_sNV ./ Δtₙ .- kron(A, ∇F)
        Z[ind_Vⱼ, ind_Vⱼ] .= dfmom

        # Determine LU decomposition
        # Z_fact = factorize(Z)
        Z_fact = lu(Z)
    end

    while maximum(abs.(f)) > abstol
        if newton_type == :approximate
            # Approximate Newton
            # ΔQⱼ = Z \ fⱼ

            # Re-use the decomposition
            # ΔQⱼ = Z_fact \ fⱼ
            ldiv!(ΔQⱼ, Z_fact, fⱼ)
        elseif newton_type == :full
            # Full Newton
            momentum_allstage!(
                Fⱼ,
                ∇Fⱼ,
                Vⱼ,
                Vⱼ,
                pⱼ,
                tⱼ,
                setup,
                cache,
                momentum_cache;
                getJacobian = true,
            )

            # Update iteration matrix
            mul!(dfmom, A_ext, ∇Fⱼ)
            dfmom = Ω_sNV / Δtₙ - dfmom
            # dfmom .= Ω_sNV / Δtₙ - A_ext * ∇Fⱼ
            Z[ind_Vⱼ, ind_Vⱼ] .= dfmom

            # Get change
            ΔQⱼ = Z \ fⱼ
        end

        # Update solution vector
        Qⱼ .+= ΔQⱼ

        # Update iteration counter
        iter += 1

        # Evaluate RHS for next iteration and check residual based on computed Vⱼ, pⱼ
        momentum_allstage!(Fⱼ, ∇Fⱼ, Vⱼ, Vⱼ, pⱼ, tⱼ, setup, cache, momentum_cache)
        mul!(fmomⱼ, A_ext, Fⱼ)
        @. fmomⱼ -= (Ωtot * Vⱼ - Ωtot * Vtotₙ) / Δtₙ
        # fmomⱼ = -(Ωtot .* Vⱼ - Ωtot .* Vtotₙ) / Δtₙ + A_ext * Fⱼ
        fmassⱼ .= yMtot
        mul!(fmassⱼ, Mtot, Vⱼ, -1, -1)
        # fmassⱼ = -(Mtot * Vⱼ + yMtot)

        iter ≤ maxiter || error("Newton solver not converged in $maxiter iterations")
    end

    # Solution at new time step with b-coefficients of RK method
    mul!(V, b_ext, Fⱼ)
    @. V = Vₙ + Δtₙ * Ω⁻¹ * V
    # V .= Vₙ .+ Δtₙ .* Ω⁻¹ .* (b_ext * Fⱼ)

    # Make V satisfy the incompressibility constraint at n+1; this is only needed when the
    # boundary conditions are time-dependent. For stiffly accurate methods, this can also
    # be skipped (e.g. Radau IIA) - this still needs to be implemented
    if setup.bc.bc_unsteady
        # Allocates new yM
        set_bc_vectors!(setup, tₙ + Δtₙ)
        (; yM) = setup.operators
        f .= yM
        mul!(f, M, V, 1 / Δtₙ, 1 / Δtₙ)
        # f .= 1 / Δtₙ .* (M * V .+ yM)
        pressure_poisson!(pressure_solver, p, f)

        mul!(Gp, G, p)
        @. V -= Δtₙ * Ω⁻¹ * Gp

        if p_add_solve
            pressure_additional_solve!(pressure_solver, V, p, tₙ + Δtₙ, setup, momentum_cache, F, f, Δp)
        else
            # Standard method; take last pressure
            p .= pⱼ[(end - Np + 1):end]
        end
    else
        # For steady bc we do an additional pressure solve
        # That saves a pressure solve for iter = 1 in the next time step
        # pressure_additional_solve!(pressure_solver, V, p, tₙ + Δtₙ, setup, momentum_cache, F, f, Δp)

        # Standard method; take pressure of last stage
        p .= pⱼ[(end - Np + 1):end]
    end

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ

    stepper
end

"""
    momentum_allstage!(F, ∇F, V, C, p, t, setup, cache, momentum_cache; getJacobian = false)

Call momentum for multiple `(V, p)` pairs, as required in implicit RK methods.
"""
function momentum_allstage!(
    Fⱼ,
    ∇Fⱼ,
    Vⱼ,
    ϕⱼ,
    pⱼ,
    tⱼ,
    setup,
    cache,
    momentum_cache;
    getJacobian = false,
)
    (; NV, Np) = setup.grid
    (; c, ∇F) = cache

    for i = 1:length(c)
        # Indices for current stage
        ind_Vᵢ = (1:NV) .+ NV * (i - 1)
        ind_pᵢ = (1:Np) .+ Np * (i - 1)

        # Quantities at current stage
        F = @view Fⱼ[ind_Vᵢ]
        V = @view Vⱼ[ind_Vᵢ]
        ϕ = @view ϕⱼ[ind_Vᵢ]
        p = @view pⱼ[ind_pᵢ]
        t = tⱼ[i]

        # Compute residual and Jacobian for this stage
        momentum!(F, ∇F, V, ϕ, p, t, setup, momentum_cache; getJacobian)

        ∇Fⱼ[ind_Vᵢ, ind_Vᵢ] = ∇F
    end

    Fⱼ, ∇Fⱼ
end
