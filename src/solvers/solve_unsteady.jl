"""
    solve_unsteady!(solution, setup)

Main solver file for unsteady calculations
"""
function solve_unsteady(setup, V₀, p₀)
    # Setup
    @unpack is_steady, visc = setup.case
    @unpack Nu, Nv, NV, Np, Nx, Ny = setup.grid
    @unpack G, M, yM = setup.discretization
    @unpack Jacobian_type, nPicard, Newton_factor, nonlinear_acc, nonlinear_maxit =
        setup.solver_settings
    @unpack use_rom = setup.rom
    @unpack t_start, t_end, Δt, isadaptive, time_stepper, time_stepper_startup, nstartup = setup.time
    @unpack save_unsteady = setup.output
    @unpack do_rtp, rtp_n = setup.visualization

    # Initialize solution vectors
    # Loop index
    n = 0
    t = t_start
    V = copy(V₀)
    p = copy(p₀)

    # Temporary variables
    momentum_cache = MomentumCache(setup)

    # Runge Kutta intermediate stages
    stepper_cache = time_stepper_cache(time_stepper, setup)
    @unpack F = stepper_cache

    # For methods that need a velocity field at n-1 the first time step
    # (e.g. AB-CN, oneleg beta) use ERK or IRK
    if needs_startup_stepper(time_stepper)
        println("Starting up with method $(time_stepper_startup)")
        ts = time_stepper_startup
        ts_cache = time_stepper_cache(time_stepper_startup, setup)
    else
        ts = time_stepper
        ts_cache = stepper_cache
    end

    # For methods that need previous solution
    Vₙ₋₁ = copy(V)
    pₙ₋₁ = copy(p)
    cₙ₋₁ = copy(V)
    convection!(cₙ₋₁, nothing, V, V, t, setup, momentum_cache)

    # Current solution
    Vₙ = copy(V)
    pₙ = copy(p)
    tₙ = t
    Δtₙ = Δt

    # Initialize BC arrays
    set_bc_vectors!(setup, t)

    if do_rtp
        rtp = initialize_rtp(setup, V, p, t)
    end

    # Estimate number of time steps that will be taken
    nt = ceil(Int, (t_end - t_start) / Δt)

    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)
    maxres = maximum(abs.(F))

    # record(fig, "output/vorticity.mp4", 2:nt; framerate = 60) do n
    while t < t_end
        # Advance one time step
        n += 1
        if n == nstartup + 1 && needs_startup_stepper(time_stepper)
            println("n = $n: switching to primary time stepper ($time_stepper)")
            ts = time_stepper
            ts_cache = stepper_cache
        end
        Vₙ₋₁ .= Vₙ
        pₙ₋₁ .= pₙ
        Vₙ .= V
        pₙ .= p
        tₙ = t

        # Change timestep based on operators
        if isadaptive && rem(n, n_adapt_Δt) == 0
            Δtₙ = get_timestep(setup, ts)
        end
        t = tₙ + Δtₙ

        println("tₙ = $tₙ, maxres = $maxres")

        # Perform a single time step with the time integration method
        if ts isa AdamsBashforthCrankNicolsonStepper
            # Get current convection and next (V, p)
            cₙ = step!(ts, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, cₙ₋₁, tₙ, Δtₙ, setup, ts_cache, momentum_cache)

            # Update previous convection
            cₙ₋₁ .= cₙ
        else
            step!(ts, V, p, Vₙ, pₙ, Vₙ₋₁, pₙ₋₁, tₙ, Δtₙ, setup, ts_cache, momentum_cache)
        end

        # Calculate mass, momentum and energy
        maxdiv, umom, vmom, k = compute_conservation(V, t, setup)

        # Residual (in Finite Volume form)
        # For ke model residual also contains k and e terms and is computed in solver_unsteady_ke
        if use_rom
            # Get ROM residual
            F, = momentum_rom(R, 0, t, setup)
            maxres = maximum(abs.(F))
        else
            if visc != "keps"
                # Norm of residual
                momentum!(F, nothing, V, V, p, t, setup, momentum_cache)
                maxres = maximum(abs.(F))
            end
        end

        if do_rtp && mod(n, rtp_n) == 0
            update_rtp!(rtp, setup, V, p, t)
        end
    end

    V, p
end
