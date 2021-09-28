"""
    step_ERK()

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong manner via the `ubc` and `vbc` functions.
"""
function step_ERK(Vₙ, pₙ, tₙ, Δt, setup)
    @unpack Nu, Nv, Np, Om_inv = setup.grid
    @unpack G, M, yM = setup.discretization

    ## get coefficients of RK method
    A_RK, b_RK, c_RK, = tableau(setup.time.rk_method)

    # number of stages
    nstage = length(b_RK)

    # we work with the following "shifted" Butcher tableau, because A_RK[1, 1]
    # is always zero for explicit methods
    A_RK = [A_RK[2:end, :]; b_RK']

    # vector with time instances
    c_RK = [c_RK[2:end]; 1] # 1 is the time level of final step

    ## preprocessing
    # store variables at start of time step
    V = Vₙ
    p = pₙ

    # right hand side evaluations, initialized at zero
    kV = zeros(Nu + Nv, nstage)

    # array for the pressure
    kp = zeros(Np, nstage)

    if setup.bc.bc_unsteady
        setup = set_bc_vectors(tₙ, setup)
    end

    yMn = yM

    tᵢ = tₙ

    ## start looping over stages

    # at i = 1 we calculate F_1, p_2 and u_2
    # ⋮
    # at i = s we calculate F_s, p_(n+1) and u_(n+1)
    for i_RK = 1:nstage
        # right-hand side for tᵢ based on current velocity field uh, vh at
        # level i
        # this includes force evaluation at tᵢ and pressure gradient
        # boundary conditions will be set through set_bc_vectors inside F
        # the pressure p is not important here, it will be removed again in the
        # next step
        _, F_rhs = momentum(V, V, p, tᵢ, setup)

        # store right-hand side of stage i
        # by adding G*p we effectively REMOVE the pressure contribution Gx*p and Gy*p (but not the
        # vectors y_px and y_py)
        kV[:, i_RK] = Om_inv .* (F_rhs + G * p)

        # update velocity current stage by sum of F_i"s until this stage,
        # weighted with Butcher tableau coefficients
        # this gives uᵢ₊₁, and for i=s gives u_(n+1)
        Vtemp = kV * A_RK[i_RK, :]

        # to make the velocity field uᵢ₊₁ at tᵢ₊₁ divergence-free we need
        # the boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c_RK[i_RK] * Δt
        if setup.bc.bc_unsteady
            set_bc_vectors!(tᵢ, setup)
        end

        # divergence of intermediate velocity field is directly calculated with M
        # old formulation:
        # f = (M*Vtemp + (yM-yMn)/Δt)/c_RK[i_RK];
        # new formulation, prevents growth of constraint errors:
        # instead of -yMn we use +M*Vₙ; they are the same up to machine
        # precision but using the latter prevents error accumulation
        # note: we should have sum(f) = 0 for periodic and no-slip bc
        f = (M * (Vₙ / Δt + Vtemp) + yM / Δt) / c_RK[i_RK]

        # solve the Poisson equation for the pressure, but not for the first
        # step if the boundary conditions are steady
        if setup.bc.bc_unsteady || i_RK > 1
            # the time tᵢ below is only for output writing
            Δp = pressure_poisson(f, tᵢ, setup)
        else # bc steady AND i_RK = 1
            Δp = pₙ
        end
        # store pressure
        kp[:, i_RK] = Δp

        # update velocity current stage, which is now divergence free
        V = Vₙ + Δt * (Vtemp - c_RK[i_RK] * Om_inv .* (G * Δp))
    end

    if setup.bc.bc_unsteady
        if setup.solversettings.p_add_solve
            p = pressure_additional_solve(V, p, tₙ + Δt, setup)
        else
            # standard method
            p = kp[:, end]
        end
    else
        # for steady bc we do an additional pressure solve
        # that saves a pressure solve for i = 1 in the next time step
        p = pressure_additional_solve(V, p, tₙ + Δt, setup)
    end

    V_new = V
    p_new = p

    V_new, p_new
end
