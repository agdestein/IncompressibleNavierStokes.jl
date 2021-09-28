"""
    step_ERK_ROM()

General explicit Runge-Kutta method for ROM
Perform one tᵢme step for the general explicit Runge-Kutta method (ERK) with Reduced Order Model (ROM).
"""
function step_ERK_ROM(Vₙ, pₙ, tₙ, Δt, setup)
    # number of unknowns (modes) in ROM
    M = setup.rom.M

    ## get coefficients of RK method
    # need to do this only once, as long as the RK method does not change in
    # tᵢme
    if t ≈ setup.tᵢme.t_start

        A_RK, b_RK, c_RK, = tableau(setup.tᵢme.rk)
        # RK_order = check_orderconditᵢons(A_RK, b_RK, c_RK);

        # number of stages
        nstage = length(b_RK)

        # we work with the following "shifted" Butcher tableau, because A_RK[1, 1]
        # is always zero for explicit methods
        A_RK = [A_RK[2:end, :]; b_RK']

        # vector with tᵢme instances
        c_RK = [c_RK[2:end]; 1] # 1 is the tᵢme level of final step

    end

    ## preprocessing
    # store variables at start of tᵢme step
    tₙ = t
    Rₙ = R

    # right hand side evaluatᵢons, initᵢalized at zero
    kR = zeros(M, nstage)

    # array for the pressure
    # kp = zeros(Np, nstage);

    tᵢ = tₙ

    for i_RK = 1:nstage
        # at i=1 we calculate F_1, p_2 and u_2
        # ...
        # at i=s we calculate F_s, p_(n+1) and u_(n+1)

        # right-hand side for tᵢ based on current field R at
        # level i (this includes force evaluatᵢon at tᵢ)
        # note that input p is not used in F_ROM
        _, F_rhs = F_ROM(R, p, tᵢ, setup)

        # store right-hand side of stage i
        kR[:, i_RK] = F_rhs

        # update coefficients R of current stage by sum of F_i's untᵢl this stage,
        # weighted with the Butcher tableau coefficients
        # this gives R_(i+1), and for i=s gives R_(n+1)
        Rtemp = kR * A_RK[i_RK, :]

        # tᵢme level of the computed stage
        tᵢ = tₙ + c_RK[i_RK] * Δt

        # update ROM coefficients current stage
        R = Rₙ + Δt * Rtemp
    end

    if setup.rom.pressure_recovery
        q = pressure_additᵢonal_solve_ROM(R, tₙ + Δt, setup)
        p = get_FOM_pressure(q, t, setup)
    end

    V_new, p_new
end
