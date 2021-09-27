"""
Check input.
"""
function check_input!(setup)
    if order4
        if visc != "laminar"
            error(
                "order 4 only implemented for laminar flow; need to add Su_vx and Sv_uy for 4th order method",
            )
        end

        if regularization != "no"
            error("order 4 only implemented for standard convection with regularization == \"no\"")
        end
    end

    # initialize solution vectors
    # loop index
    n = 1

    # initial velocity field
    # uh = u_start[:]
    # vh = v_start[:]
    V = V_start # [uh; vh]
    V_old = V

    if visc == "keps"
        kth = kt[:]
        eh = e[:]
    end

    # for unsteady problems allocate k, umom and vmom, maxdiv and time
    # for steady problems the time for allocating during running is negligible,
    # since normally only a few iterations are required
    if !steady
        t = t_start
        if timestep.set
            dt = set_timestep(setup)
        end

        # Estimate number of time steps that will be taken
        nt = ceil((t_end - t_start) / dt)

        # allocate variables, including initial condition
        maxres = zeros(nt + 1)
        maxdiv = zeros(nt + 1)
        k = zeros(nt + 1)
        umom = zeros(nt + 1)
        vmom = zeros(nt + 1)
        time = zeros(nt + 1)
        nonlinear_its = zeros(nt + 1)
    end

    # kinetic energy and momentum of initial velocity field
    # iteration 1 corresponds to t = 0 (for unsteady simulations)
    maxdiv[1], umom[1], vmom[1], k[1] = check_conservation(V, t, setup)

    if maxdiv[1] > 1e-12 && !steady
        @warn "Initial velocity field not (discretely) divergence free: $(maxdiv[1])"
        fprintf(
            fcw,
            "additional projection to make initial velocity field divergence free\n",
        )

        # make velocity field divergence free
        Om_inv = setup.grid.Om_inv
        G = setup.discretization.G
        Nu = setup.grid.Nu
        Nv = setup.grid.Nv

        f = setup.discretization.M * V + setup.discretization.yM
        dp = pressure_poisson(f, t, setup)
        V -= Om_inv .* (G * dp)
        # uh = V[1:Nu]
        # vh = V[Nu+1:end]
        # repeat conservation with updated velocity field
        maxdiv[1], umom[1], vmom[1], k[1] = check_conservation(V, t, setup)
    end

    symmetry_flag, symmetry_error = check_symmetry(V, t, setup)

    # initialize pressure
    if !setup.rom.use_rom
        if setup.case.steady
            # for steady state computations, the initial guess is the provided initial condition
            p = p_start[:]
        else
            if setup.solver_settings.p_initial
                # calculate initial pressure from a Poisson equation
                p = pressure_additional_solve(V, p_start[:], t, setup)
            else
                # use provided initial condition (not recommended)
                p = p_start[:]
            end
        end
    end

    # ROM: uses the IC for the pressure; note that in solver_unsteady the pressure will be
    # computed from the ROM PPE, after the ROM basis has been set-up
    if setup.rom.use_rom
        if setup.case.steady
            error("ROM not implemented for steady flow")
        else
            p = p_start[:]
        end
    end

    # for steady problems, with Newton linearization and full Jacobian, first start with nPicard Picard iterations
    if setup.case.steady
        setup.solver_settings.Newton_factor = false
    elseif method == 21 || (exist("method_startup", "var") && method_startup == 21)
        # implicit RK time integration
        setup.solver_settings.Newton_factor = true
    end

    # residual of momentum equations at start
    maxres[1], _ = momentum(V, V, p, t, setup, false)

    if !steady && save_unsteady
        # allocate space for variables
        uh_total = zeros(nt, setup.grid.Nu)
        vh_total = zeros(nt, setup.grid.Nv)
        p_total = zeros(nt, setup.grid.Np)

        # store initial solution
        uh_total[1, :] = u_start[:]
        vh_total[1, :] = v_start[:]
        p_total[1, :] = p_start[:]
    end

    if !rtp.show && rtp.movie
        warning(
            "real-time plotting off (rtp.show = 0) but movie generation is on (rtp.movie = true). no movie will be generated.",
        )
    end

    if !steady && save_unsteady && !save_file
        @warn "unsteady data is stored in workspace (save_unsteady = true) but not saved to a file (save_file = false)"
    end

    setup
end
