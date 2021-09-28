## general implicit Runge-Kutta method

# (unsteady) Dirichlet boundary points are not part of solution vector but
# are prescribed in a "strong" manner via the ubc and vbc functions
function step_IRK(Vₙ, pₙ, tₙ, Δt, setup)


    ## grid info
    Nu = setup.grid.Nu
    Nv = setup.grid.Nv
    NV = Nu + Nv
    Np = setup.grid.Np
    Om_inv = setup.grid.Om_inv
    Om = setup.grid.Om

    G = setup.discretization.G
    M = setup.discretization.M
    yM = setup.discretization.yM

    yMn = yM

    # get coefficients of RK method
    A_RK, b_RK, c_RK, = tableau(setup.time.rk_method)

    # Number of stages
    nstage = length(b_RK)

    setup.time.A_RK = A_RK
    setup.time.b_RK = b_RK
    setup.time.c_RK = c_RK
    setup.time.nstage = nstage

    # extend the Butcher tableau
    Is = sparse(I, nstage, nstage)
    Om_sNV = kron(Is, spdiagm(Om))
    A_RK_ext = kron(A_RK, sparse(I, NV, NV))
    b_RK_ext = kron(b_RK', sparse(I, NV, NV))
    c_RK_ext = spdiagm(c_RK)

    ## preprocessing

    # store variables at start of time step
    # tₙ = t;
    # Vₙ = V;
    # pₙ = p;
    # qn = [Vₙ; pₙ];
    p = pₙ

    # tj contains the time instances at all stages, tj = [t1;t2;...;ts]
    tj = tₙ + c_RK * Δt

    # gradient operator
    Gtot = kron(A_RK, G) # could also use 1 instead of c_RK and later scale the pressure

    # divergence operator
    Mtot = kron(Is, M)

    # finite volumes
    Omtot = kron(ones(nstage), Om)

    # boundary condition for divergence operator
    if setup.bc.bc_unsteady
        set_bc_vectors!(tₙ, setup)
    end

    # to make the velocity field u_(i+1) at t_(i+1) divergence-free we need
    # the boundary conditions at t_(i+1)
    if setup.bc.bc_unsteady
        yMtot = zeros(Np, nstage)
        for i = 1:nstage
            ti = tj[i]
            setup = set_bc_vectors(ti, setup)
            yMtot[:, i] = setup.discretization.yM
        end
        yMtot = yMtot[:]
    else
        yMtot = kron(ones(nstage), yMn)
    end

    # zero block in iteration matrix
    Z2 = spzeros(nstage * Np, nstage * Np)

    # iteration counter
    i = 0

    # iteration error
    nonlinear_maxit = setup.solversettings.nonlinear_maxit
    error_nonlinear = zeros(nonlinear_maxit)

    # Vtot contains all stages and is ordered as [u1;v1;u2;v2;...;us;vs];
    # initialize with the solution at tₙ
    Vtotₙ = kron(ones(nstage), Vₙ)
    ptotₙ = kron(ones(nstage), pₙ)

    # index in global solution vector
    indxV = 1:NV*nstage
    indxp = (NV*nstage+1):(NV+Np)*nstage

    # starting guess for intermediate stages => this can be improved, see e.g.
    # the Radau, Gauss4, or Lobatto scripts
    Vⱼ = Vtotₙ
    pⱼ = ptotₙ
    Qⱼ = [Vⱼ; pⱼ]

    # initialize right-hand side for all stages
    _, F_rhs, = F_multiple(Vⱼ, Vⱼ, pⱼ, tj, setup, false)
    # initialize momentum residual
    fmom = -(Omtot .* Vⱼ - Omtot .* Vtotₙ) / Δt + A_RK_ext * F_rhs
    # initialize mass residual
    fmass = -(Mtot * Vⱼ + yMtot)
    f = [fmom; fmass]

    if setup.solversettings.nonlinear_Newton == "approximate"
        # approximate Newton
        # Jacobian based on current solution un
        _, _, Jn = F(Vₙ, Vₙ, pₙ, tₙ, setup, true)
        # form iteration matrix, which is now fixed during iterations
        dfmom = Om_sNV / Δt - kron(A_RK, Jn)
        #
        Z = [dfmom Gtot; Mtot Z2]

        # determine LU decomposition; often this is too slow
        #     [L, U] = lu(Z);
    end

    while maximum(abs.(f)) > setup.solversettings.nonlinear_acc

        if setup.solversettings.nonlinear_Newton == "approximate"
            # approximate Newton
            # do not rebuild Z
            ΔQⱼ = Z \ f

            # re-use the LU decomposition (often too slow):
            # ΔQⱼ = U\(L\f);
        elseif setup.solversettings.nonlinear_Newton == "full"
            # full Newton
            _, _, J = F_multiple(Vⱼ, Vⱼ, pⱼ, tj, setup, true)

            # form iteration matrix
            dfmom = Om_sNV / Δt - A_RK_ext * J
            Z = [dfmom Gtot; Mtot Z2]

            # get change
            ΔQⱼ = Z \ f
        end

        # update solution vector
        Qⱼ .+= ΔQⱼ
        Vⱼ = @view Qⱼ[indxV]
        pⱼ = @view Qⱼ[indxp]

        # update iteration counter
        i = i + 1

        # evaluate rhs for next iteration and check residual based on
        # computed Vⱼ, pⱼ
        _, F_rhs, = F_multiple(Vⱼ, Vⱼ, pⱼ, tj, setup, false)
        fmom = -(Omtot .* Vⱼ - Omtot .* Vtotₙ) / Δt + A_RK_ext * F_rhs
        fmass = -(Mtot * Vⱼ + yMtot)

        f = [fmom; fmass]

        error_nonlinear[i] = maximum(abs.(f))
        if i > nonlinear_maxit
            error("Newton not converged in $nonlinear_maxit iterations")
        end

    end

    # store number of iterations
    iterations = i

    # solution at new time step with b-coefficients of RK method
    V = Vₙ + Δt * Om_inv .* (b_RK_ext * F_rhs)

    # make V satisfy the incompressibility constraint at n+1; this is only
    # needed when the boundary conditions are time-dependent
    # for stiffly accurate methods, this can also be skipped (e.g. Radau IIA) -
    # this still needs to be implemented
    if setup.bc.bc_unsteady
        setup = set_bc_vectors(tₙ + Δt, setup)
        f = 1 / Δt * (M * V + yM)
        Δp = pressure_poisson(f, tₙ + Δt, setup)
        V .-= Δt * Om_inv .* (G * Δp)
    end

    if setup.bc.bc_unsteady
        if setup.solversettings.p_add_solve
            p = pressure_additional_solve(V, p, tₙ + Δt, setup)
        else
            # standard method; take last pressure
            p = pⱼ[end-Np+1:end]
        end
    else
        # for steady bc we do an additional pressure solve
        # that saves a pressure solve for i = 1 in the next time step
        # p = pressure_additional_solve(V, p, tₙ+Δt, setup);
        # standard method; take pressure of last stage
        p = pⱼ[end-Np+1:end]
    end

    V_new = V
    p_new = p

    V_new, p_new, iterations
end
