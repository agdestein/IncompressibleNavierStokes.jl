"""
Check input.
"""
function check_input!(setup, V_start, p_start, t)
    @unpack is_steady, visc = setup.case
    @unpack order4, G, M, yM = setup.discretization
    @unpack Ω⁻¹, NV = setup.grid
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

    # Initialize solution vectors
    # Loop index
    n = 1

    # Initial velocity field
    V = copy(V_start) # [uₕ; vₕ]

    if visc == "keps"
        kth = kt[:]
        eh = e[:]
    end

    # For unsteady problems allocate k, umom and vmom, maxdiv and time
    # For steady problems the time for allocating during running is negligible,
    # Since normally only a few iterations are required
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


    # Allocate variables, including initial condition
    maxres = zeros(nt + 1)
    maxdiv = zeros(nt + 1)
    k = zeros(nt + 1)
    umom = zeros(nt + 1)
    vmom = zeros(nt + 1)
    time = zeros(nt + 1)
    nonlinear_its = zeros(nt + 1)

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t = 0 (for unsteady simulations)
    maxdiv[1], umom[1], vmom[1], k[1] = check_conservation(V, t, setup)

    if maxdiv[1] > 1e-12 && !is_steady
        @warn "Initial velocity field not (discretely) divergence free: $(maxdiv[1])"
        println("Additional projection to make initial velocity field divergence free")

        # Make velocity field divergence free
        f = M * V + yM
        Δp = pressure_poisson(f, t, setup)
        V .-= Ω⁻¹ .* (G * Δp)

        # Repeat conservation with updated velocity field
        maxdiv[1], umom[1], vmom[1], k[1] = check_conservation(V, t, setup)
    end

    symmetry_flag, symmetry_error = check_symmetry(V, t, setup)

    # Initialize pressure
    p = p_start[:]
    if setup.rom.use_rom
        # ROM: uses the IC for the pressure; note that in solver_unsteady the pressure will be
        # Computed from the ROM PPE, after the ROM basis has been set-up
        if setup.case.is_steady
            error("ROM not implemented for steady flow")
        end
    else
        if setup.case.is_steady
            # For steady state computations, the initial guess is the provided initial condition
        else
            if setup.solver_settings.p_initial
                # Calculate initial pressure from a Poisson equation
                pressure_additional_solve!(V, p, t, setup)
            else
                # Use provided initial condition (not recommended)
            end
        end
    end

    # For steady problems, with Newton linearization and full Jacobian, first start with nPicard Picard iterations
    if setup.case.is_steady
        setup.solver_settings.Newton_factor = false
    elseif method == 21 || method_startup == 21
        # Implicit RK time integration
        setup.solver_settings.Newton_factor = true
    end

    # Residual of momentum equations at start
    F, = momentum(V, V, p, t, setup, false)
    maxres[1] = maximum(abs.(F))

    if !is_steady && save_unsteady
        # Allocate space for variables
        uₕ_total = zeros(nt, setup.grid.Nu)
        vₕ_total = zeros(nt, setup.grid.Nv)
        p_total = zeros(nt, setup.grid.Np)

        # Store initial solution
        uₕ_total[1, :] = u_start[:]
        vₕ_total[1, :] = v_start[:]
        p_total[1, :] = p_start[:]
    end

    (; V, p, t, Δt, n, nt, maxres, maxdiv, k, vmom, nonlinear_its, time, umom)
end
