"""
    step_ERK()

Perform one time step for the general explicit Runge-Kutta method (ERK).

Dirichlet boundary points are not part of solution vector but are prescribed in a strong manner via the `ubc` and `vbc` functions.
"""
function step_ERK!(V, p, Vₙ, pₙ, tₙ, Δt, setup)
    @unpack Nu, Nv, Np, Om_inv = setup.grid
    @unpack G, M, yM = setup.discretization

    ## get coefficients of RK method
    A, b, c, = tableau(setup.time.rk_method)

    # number of stages
    nstage = length(b)

    # we work with the following "shifted" Butcher tableau, because A[1, :]
    # is always zero for explicit methods
    A = [A[2:end, :]; b']

    # vector with time instances (1 is the time level of final step)
    c = [c[2:end]; 1]

    ## preprocessing
    # store variables at start of time step
    V .= Vₙ
    p .= pₙ

    # right hand side evaluations, initialized at zero
    kV = zeros(Nu + Nv, nstage)

    # array for the pressure
    kp = zeros(Np, nstage)

    if setup.bc.bc_unsteady
        set_bc_vectors!(setup, tₙ)
    end

    tᵢ = tₙ

    ## start looping over stages

    # at i = 1 we calculate F₁, p₂ and u₂
    # ⋮
    # at i = s we calculate Fₛ, pₙ₊₁, and uₙ₊₁
    for i = 1:nstage
        # Right-hand side for tᵢ based on current velocity field uₕ, vₕ at, level i
        # this includes force evaluation at tᵢ and pressure gradient
        # boundary conditions will be set through set_bc_vectors! inside momentum
        # the pressure p is not important here, it will be removed again in the
        # next step
        F_rhs = momentum(V, V, p, tᵢ, setup)

        # Store right-hand side of stage i
        # by adding G*p we effectively REMOVE the pressure contribution Gx*p and Gy*p (but not the
        # vectors y_px and y_py)
        kV[:, i] = Om_inv .* (F_rhs + G * p)

        # Update velocity current stage by sum of Fᵢ's until this stage,
        # weighted with Butcher tableau coefficients
        # this gives uᵢ₊₁, and for i=s gives uᵢ₊₁
        Vtemp = kV * A[i, :]

        # To make the velocity field uᵢ₊₁ at tᵢ₊₁ divergence-free we need
        # the boundary conditions at tᵢ₊₁
        tᵢ = tₙ + c[i] * Δt
        if setup.bc.bc_unsteady
            set_bc_vectors!(setup, tᵢ)
        end

        # Divergence of intermediate velocity field is directly calculated with M
        f = (M * (Vₙ / Δt + Vtemp) + yM / Δt) / c[i]

        # Solve the Poisson equation for the pressure, but not for the first
        # step if the boundary conditions are steady
        if setup.bc.bc_unsteady || i > 1
            # The time tᵢ below is only for output writing
            Δp = pressure_poisson(f, tᵢ, setup)
        else
            # bc steady AND i = 1
            Δp = pₙ
        end

        # Store pressure
        kp[:, i] = Δp

        # Update velocity current stage, which is now divergence free
        V .= Vₙ .+ Δt .* (Vtemp .- c[i] .* Om_inv .* (G * Δp))
    end

    if setup.bc.bc_unsteady
        if setup.solversettings.p_add_solve
            pressure_additional_solve!(V, p, tₙ + Δt, setup)
        else
            # Standard method
            p .= kp[:, end]
        end
    else
        # For steady bc we do an additional pressure solve
        # that saves a pressure solve for i = 1 in the next time step
        pressure_additional_solve!(V, p, tₙ + Δt, setup)
    end

    V, p
end
