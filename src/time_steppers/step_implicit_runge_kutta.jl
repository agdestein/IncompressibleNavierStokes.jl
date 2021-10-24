"""
    step!(stepper::ImplicitRungeKuttaStepper, Δt)

Do one time step for implicit Runge-Kutta method.

Unsteady Dirichlet boundary points are not part of solution vector but
are prescribed in a "strong" manner via the `u_bc` and `v_bc` functions.
"""
function step!(stepper::ImplicitRungeKuttaStepper, Δt)
    @unpack V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache =
        stepper
    @unpack Nu, Nv, NV, Np, Ω, Ω⁻¹ = setup.grid
    @unpack G, M = setup.discretization
    @unpack pressure_solver, nonlinear_maxit, nonlinear_acc, nonlinear_Newton, p_add_solve =
        setup.solver_settings
    @unpack Vtotₙ, ptotₙ, Vⱼ, pⱼ, Qⱼ, Fⱼ, ∇Fⱼ, f, Δp = cache
    @unpack A, b, c, s, Is, Ω_sNV, A_ext, b_ext, c_ext = cache
   
    # Update current solution (does not depend on previous step size)
    stepper.n += 1
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt

    # Time instances at all stages, tⱼ = [t₁, t₂, ..., tₛ]
    tⱼ = tₙ + c * Δtₙ

    # Gradient operator (could also use 1 instead of c and later scale the pressure)
    Gtot = kron(A, G)

    # Divergence operator
    Mtot = kron(Is, M)

    # Finite volumes
    Ωtot = kron(ones(s), Ω)

    # Boundary condition for divergence operator
    if setup.bc.bc_unsteady
    end

    # Make the velocity field uᵢ₊₁ at tᵢ₊₁ divergence-free (BC at tᵢ₊₁ needed)
    if setup.bc.bc_unsteady
        yMtot = zeros(Np, s)
        for i = 1:s
            tᵢ = tⱼ[i]

            # Modifies `yM`!
            set_bc_vectors!(setup, tᵢ)
            @unpack yM = setup.discretization
            yMtot[:, i] = yM
        end
        yMtot = yMtot[:]
    else
        set_bc_vectors!(setup, tₙ)
        @unpack yM = setup.discretization
        yMtot = kron(ones(s), yM)
    end

    # Iteration counter
    iter = 0

    # Index in global solution vector
    ind_Vⱼ = 1:(NV * s)
    ind_pⱼ = (NV * s + 1):((NV + Np) * s)

    # Zero block in iteration matrix
    Z2 = spzeros(s * Np, s * Np)

    # Vtot contains all stages and is ordered as [u₁; v₁; u₂; v₂; ...; uₛ; vₛ];
    # Initialize with the solution at tₙ
    Vtotₙ = kron(ones(s), Vₙ)
    ptotₙ = kron(ones(s), pₙ)

    # Starting guess for intermediate stages
    # This can be improved, see e.g. the Radau, Gauss4, or Lobatto scripts
    Vⱼ .= Vtotₙ
    pⱼ .= ptotₙ
    Qⱼ = [Vⱼ; pⱼ]

    # Initialize right-hand side for all stages
    momentum_allstage!(Fⱼ, ∇Fⱼ, Vⱼ, Vⱼ, pⱼ, tⱼ, setup, cache, momentum_cache)

    # Initialize momentum residual
    fmom = -(Ωtot .* Vⱼ - Ωtot .* Vtotₙ) / Δtₙ + A_ext * Fⱼ

    # Initialize mass residual
    fmass = -(Mtot * Vⱼ + yMtot)
    f = [fmom; fmass]

    if nonlinear_Newton == "approximate"
        # Approximate Newton (Jacobian is based on current solution Vₙ)
        momentum!(Fₙ, ∇Fₙ, Vₙ, Vₙ, pₙ, tₙ, setup, momentum_cache; getJacobian = true)

        # Form iteration matrix, which is now fixed during iterations
        dfmom = Ω_sNV / Δtₙ - kron(A, ∇Fₙ)
        Z = [dfmom Gtot; Mtot Z2]

        # Determine LU decomposition
        Z_fact = factorize(Z)
    end

    while maximum(abs.(f)) > nonlinear_acc
        if nonlinear_Newton == "approximate"
            # Approximate Newton
            # ΔQⱼ = Z \ f

            # Re-use the decomposition
            ΔQⱼ = Z_fact \ f
        elseif nonlinear_Newton == "full"
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

            # Form iteration matrix
            dfmom = Ω_sNV / Δtₙ - A_ext * ∇Fⱼ
            Z = [dfmom Gtot; Mtot Z2]

            # Get change
            ΔQⱼ = Z \ f
        end

        # Update solution vector
        Qⱼ .+= ΔQⱼ
        Vⱼ = @view Qⱼ[ind_Vⱼ]
        pⱼ = @view Qⱼ[ind_pⱼ]

        # Update iteration counter
        iter += 1

        # Evaluate RHS for next iteration and check residual based on computed Vⱼ, pⱼ
        momentum_allstage!(Fⱼ, ∇Fⱼ, Vⱼ, Vⱼ, pⱼ, tⱼ, setup, cache, momentum_cache)
        fmom = -(Ωtot .* Vⱼ - Ωtot .* Vtotₙ) / Δtₙ + A_ext * Fⱼ
        fmass = -(Mtot * Vⱼ + yMtot)

        f = [fmom; fmass]

        if iter > nonlinear_maxit
            error("Newton not converged in $nonlinear_maxit iterations")
        end
    end

    # Solution at new time step with b-coefficients of RK method
    V .= Vₙ .+ Δtₙ .* Ω⁻¹ .* (b_ext * Fⱼ)

    # Make V satisfy the incompressibility constraint at n+1; this is only
    # needed when the boundary conditions are time-dependent.
    # For stiffly accurate methods, this can also be skipped (e.g. Radau IIA) -
    # This still needs to be implemented
    if setup.bc.bc_unsteady
        # Allocates new yM
        set_bc_vectors!(setup, tₙ + Δtₙ)
        @unpack yM = setup.discretization
        f = 1 / Δtₙ * (M * V + yM)
        pressure_poisson!(pressure_solver, Δp, f, tₙ + Δtₙ, setup)

        V .-= Δtₙ .* Ω⁻¹ .* (G * Δp)

        if p_add_solve
            pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F)
        else
            # Standard method; take last pressure
            p .= pⱼ[(end - Np + 1):end]
        end
    else
        # For steady bc we do an additional pressure solve
        # That saves a pressure solve for iter = 1 in the next time step
        # pressure_additional_solve!(V, p, tₙ + Δtₙ, setup, momentum_cache, F)

        # Standard method; take pressure of last stage
        p = pⱼ[(end - Np + 1):end]
    end

    t = tₙ + Δtₙ
    @pack! stepper = t, tₙ, Δtₙ, iter 

    stepper
end

"""
    momentum_allstage!(F, ∇F, V, C, p, t, setup, cache, momentum_cache; getJacobian = false)

Call momentum for multiple `(V, p)` pairs, as required in implicit RK methods.
"""
function momentum_allstage!(
    F,
    ∇F,
    V,
    C,
    p,
    t,
    setup,
    cache,
    momentum_cache;
    getJacobian = false,
)
    @unpack Nu, Nv, NV, Np = setup.grid
    @unpack s, c = cache

    for i = 1:s
        # Indices for current stage
        ind_Vᵢ = (1:NV) .+ NV * (i - 1)
        ind_pᵢ = (1:Np) .+ Np * (i - 1)

        # Quantities at current stage
        Fᵢ = @view F[ind_Vᵢ]
        ∇Fᵢ = @view ∇F[ind_Vᵢ, ind_Vᵢ]
        Vᵢ = @view V[ind_Vᵢ]
        Cᵢ = @view C[ind_Vᵢ]
        pᵢ = @view p[ind_pᵢ]
        tᵢ = t[i]

        # Compute residual and Jacobian for this stage
        momentum!(Fᵢ, ∇Fᵢ, Vᵢ, Cᵢ, pᵢ, tᵢ, setup, momentum_cache; getJacobian)
    end

    F, ∇F
end
