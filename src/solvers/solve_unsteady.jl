"""
    solve(unsteady_problem, setup, V₀, p₀)

Solve `unsteady_problem`.
"""
function solve(::UnsteadyProblem, setup, V₀, p₀)
    # Setup
    @unpack model, processors = setup
    @unpack use_rom = setup.rom
    @unpack t_start, t_end, Δt, isadaptive, method, method_startup, nstartup = setup.time

    # For methods that need a velocity field at n-1 the first time step
    # (e.g. AB-CN, oneleg beta) use ERK or IRK
    if needs_startup_method(method)
        println("Starting up with method $(method_startup)")
        method_use = method_startup
    else
        method_use = method
    end

    stepper = TimeStepper(method_use, setup, V₀, p₀)

    # Initialize BC arrays
    set_bc_vectors!(setup, stepper.t)

    # Processors for iteration results  
    for ps ∈ processors
        initialize!(ps, stepper)
        process!(ps, stepper)
    end

    # record(fig, "output/vorticity.mp4", 1:nt; framerate = 60) do n
    while stepper.t < t_end
        if stepper.n == nstartup && needs_startup_method(method)
            println("n = $(stepper.n): switching to primary ODE method ($method)")
            method_use = method
            stepper = change_time_stepper(stepper, method)
        end

        # Change timestep based on operators
        if isadaptive && rem(stepper.n, n_adapt_Δt) == 0
            Δt = get_timestep(setup, method_use)
        end

        # Perform a single time step with the time integration method
        step!(stepper, Δt)

        # Process iteration results with each processor
        for ps ∈ processors
            # Only update each `nupdate`-th iteration
            stepper.n % ps.nupdate == 0 && process!(ps, stepper)
        end
    end

    finalize!.(processors)

    @unpack V, p = stepper
    V, p
end
