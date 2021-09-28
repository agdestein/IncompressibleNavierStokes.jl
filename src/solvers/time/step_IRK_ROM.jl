function step_IRK_ROM(Vₙ, pₙ, tₙ, Δt, setup)
    ## general implicit Runge-Kutta method for ROM

    # number of unknowns (modes) in ROM
    M = setup.rom.M

    ## get coefficients of RK method
    if t ≈ setup.time.t_start
        A_RK, b_RK, c_RK, = tableau(setup.time.rk_method)
        # RK_order = check_orderconditions(A_RK, b_RK, c_RK);
        # number of stages
        nstage = length(b_RK)

        setup.time.A_RK = A_RK
        setup.time.b_RK = b_RK
        setup.time.c_RK = c_RK
        setup.time.nstage = nstage

        # extend the Butcher tableau
        Is = sparse(I, nstage, nstage)
        Om_sM = kron(Is, spdiagm(ones(M)))
        A_RK_ext = kron(A_RK, sparse(I, M, M))
        b_RK_ext = kron(b_RK', sparse(I, M, M))
        c_RK_ext = spdiagm(c_RK)
    end

    ## preprocessing

    # store variables at start of time step
    tₙ = t
    Rₙ = R

    # tⱼ contains the time instances at all stages, tⱼ = [t1;t2;...;ts]
    tⱼ = tₙ + c_RK * Δt

    # iteration counter
    i = 0
    # iteration error
    error_nonlinear = zeros(nonlinear_maxit)

    # Vtot contains all stages and is ordered as [u1;v1;u2;v2;...;us;vs];
    # initialize with the solution at tₙ
    Rtotₙ = kron(ones(nstage), Rₙ)

    # index in global solution vector
    indxR = 1:M*nstage

    # starting guess for intermediate stages
    Rⱼ = Rtotₙ

    Qⱼ = Rⱼ

    # initialize right-hand side for all stages
    _, F_rhs, = F_multiple_ROM(Rⱼ, [], tⱼ, setup, false)
    # initialize momentum residual
    fmom = -(Rⱼ - Rtotₙ) / Δt + A_RK_ext * F_rhs
    # initialize residual
    f = fmom

    if setup.solversettings.nonlinear_Newton == "approximate"
        # approximate Newton
        # Jacobian based on current solution un
        _, _, Jn = F_ROM(Rₙ, [], tₙ, setup, true)
        # form iteration matrix, which is now fixed during iterations
        dfmom = Om_sM / Δt - kron(A_RK, Jn)
        Z = dfmom
    end

    while maximum(abs.(f)) > setup.solversettings.nonlinear_acc
        if setup.solversettings.nonlinear_Newton == "approximate"
            # approximate Newton
            # do not rebuild Z
            ΔQⱼ = Z \ f
        elseif setup.solversettings.nonlinear_Newton == "full"
            # full Newton
            _, _, J = F_multiple_ROM(Rⱼ, [], tⱼ, setup, true)
            # form iteration matrix
            dfmom = Om_sM / Δt - A_RK_ext * J

            Z = dfmom

            # get change
            ΔQⱼ = Z \ f
        end

        # update solution vector
        Qⱼ = Qⱼ + ΔQⱼ
        Rⱼ = Qⱼ[indxR]

        # update iteration counter
        i = i + 1

        # evaluate rhs for next iteration and check residual based on
        # computed Rⱼ
        _, F_rhs, = F_multiple_ROM(Rⱼ, [], tⱼ, setup, 0)
        fmom = -(Rⱼ - Rtotₙ) / Δt + A_RK_ext * F_rhs

        f = fmom

        error_nonlinear[i] = maximum(abs.(f))
        if i > nonlinear_maxit
            error(["Newton not converged in " num2str(nonlinear_maxit) " iterations"])
        end
    end

    nonlinear_its[n] = i

    # solution at new time step with b-coefficients of RK method
    R = Rₙ + Δt * (b_RK_ext * F_rhs)

    if setup.rom.pressure_recovery
        q = pressure_additional_solve_ROM(R, tₙ + Δt, setup)
        p = getFOM_pressure(q, t, setup)
    end

    V_new, p_new
end
