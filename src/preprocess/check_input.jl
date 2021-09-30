"""
Check input.
"""
function check_input!(setup, V_start, p_start, t)
    @unpack is_steady, visc = setup.case
    @unpack order4, G, M, yM = setup.discretization
    @unpack Om_inv, NV = setup.grid
    @unpack nonlinear_maxit = setup.solver_settings
    @unpack t_start, t_end, Δt, isadaptive, method, method_startup = setup.time
    @unpack save_unsteady = setup.output

    if order4
        if visc != "laminar"
            error(
                "order 4 only implemented for laminar flow; need to add Su_vx and Sv_uy for 4th order method",
            )
        end

        if regularization != "no"
            error(
                "order 4 only implemented for standard convection with regularization == \"no\"",
            )
        end
    end

    # initialize solution vectors
    # loop index
    n = 1

    # initial velocity field
    V = copy(V_start) # [uₕ; vₕ]

    if visc == "keps"
        kth = kt[:]
        eh = e[:]
    end

    # for unsteady problems allocate k, umom and vmom, maxdiv and time
    # for steady problems the time for allocating during running is negligible,
    # since normally only a few iterations are required
    if !is_steady
        t = t_start
        if isadaptive
            Δt = get_timestep(setup)
        end

        # Estimate number of time steps that will be taken
        nt = ceil(Int, (t_end - t_start) / Δt)
    else
        nt = nonlinear_maxit
    end

    # Construct body force or immersed boundary method
    # The body force is called in the residual routines e.g. momentum.jl
    # Steady force can be precomputed once:
    if setup.force.isforce
        setup.force.Fx, setup.force.Fy, _ = bodyforce(V, t, setup)
    else
        setup.force.Fx = zeros(NV)
        setup.force.Fy = zeros(NV)
    end


    # allocate variables, including initial condition
    maxres = zeros(nt + 1)
    maxdiv = zeros(nt + 1)
    k = zeros(nt + 1)
    umom = zeros(nt + 1)
    vmom = zeros(nt + 1)
    time = zeros(nt + 1)
    nonlinear_its = zeros(nt + 1)

    # kinetic energy and momentum of initial velocity field
    # iteration 1 corresponds to t = 0 (for unsteady simulations)
    maxdiv[1], umom[1], vmom[1], k[1] = check_conservation(V, t, setup)

    if maxdiv[1] > 1e-12 && !is_steady
        @warn "Initial velocity field not (discretely) divergence free: $(maxdiv[1])"
        println("Additional projection to make initial velocity field divergence free")

        # make velocity field divergence free
        f = M * V + yM
        Δp = pressure_poisson(f, t, setup)
        V .-= Om_inv .* (G * Δp)

        # repeat conservation with updated velocity field
        maxdiv[1], umom[1], vmom[1], k[1] = check_conservation(V, t, setup)
    end

    symmetry_flag, symmetry_error = check_symmetry(V, t, setup)

    # initialize pressure
    p = p_start[:]
    if setup.rom.use_rom
        # ROM: uses the IC for the pressure; note that in solver_unsteady the pressure will be
        # computed from the ROM PPE, after the ROM basis has been set-up
        if setup.case.is_steady
            error("ROM not implemented for steady flow")
        end
    else
        if setup.case.is_steady
            # for steady state computations, the initial guess is the provided initial condition
        else
            if setup.solver_settings.p_initial
                # calculate initial pressure from a Poisson equation
                pressure_additional_solve!(V, p, t, setup)
            else
                # use provided initial condition (not recommended)
            end
        end
    end

    # for steady problems, with Newton linearization and full Jacobian, first start with nPicard Picard iterations
    if setup.case.is_steady
        setup.solver_settings.Newton_factor = false
    elseif method == 21 || method_startup == 21
        # implicit RK time integration
        setup.solver_settings.Newton_factor = true
    end

    # residual of momentum equations at start
    F, = momentum(V, V, p, t, setup, false)
    maxres[1] = maximum(abs.(F))

    if !is_steady && save_unsteady
        # allocate space for variables
        uₕ_total = zeros(nt, setup.grid.Nu)
        vₕ_total = zeros(nt, setup.grid.Nv)
        p_total = zeros(nt, setup.grid.Np)

        # store initial solution
        uₕ_total[1, :] = u_start[:]
        vₕ_total[1, :] = v_start[:]
        p_total[1, :] = p_start[:]
    end

    (; V, p, t, Δt, n, nt, maxres, maxdiv, k, vmom, nonlinear_its, time, umom)
end
