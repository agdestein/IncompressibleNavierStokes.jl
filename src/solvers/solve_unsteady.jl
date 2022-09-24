"""
    solve(
        problem::UnsteadyProblem,
        method;
        pressure_solver = DirectPressureSolver(problem.setup),
        Δt = nothing,
        cfl = 1,
        n_adapt_Δt = 1,
        nstartup = 1,
        method_startup = nothing,
        inplace = false,
        processors = [],
    )

Solve unsteady problem using `method`.

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl` .

For methods that are not self-starting, `nstartup` startup iterations are performed with
`method_startup`.

Each `processor` is called after every `processor.nupdate` time step.
"""
function solve(
    problem::UnsteadyProblem,
    method;
    pressure_solver = DirectPressureSolver(problem.setup),
    Δt = nothing,
    cfl = 1,
    n_adapt_Δt = 1,
    nstartup = 1,
    method_startup = nothing,
    inplace = false,
    processors = [],
)
    (; setup, V₀, p₀, tlims) = problem

    t_start, t_end = tlims
    isadaptive = isnothing(Δt)
    if !isadaptive
        nstep = round(Int, (t_end - t_start) / Δt)
        Δt = (t_end - t_start) / nstep
    end

    # For methods that need a velocity field at n-1 the first time step
    # (e.g. AB-CN, oneleg beta) use ERK or IRK
    if needs_startup_method(method)
        println("Starting up with method $(method_startup)")
        method_use = method_startup
    else
        method_use = method
    end

    if inplace
        cache = ode_method_cache(method_use, setup)
        momentum_cache = MomentumCache(setup)
    end

    stepper = TimeStepper(;
        method = method_use,
        setup,
        pressure_solver,
        V = copy(V₀),
        p = copy(p₀),
        t = copy(t_start),
        Vₙ = copy(V₀),
        pₙ = copy(p₀),
        tₙ = copy(t_start),
    )
    isadaptive && (Δt = get_timestep(stepper, cfl))

    # Initialize BC arrays
    bc_vectors = get_bc_vectors(setup, stepper.t)

    # Processors for iteration results  
    for ps ∈ processors
        initialize!(ps, stepper)
        process!(ps, stepper)
    end

    while stepper.t < t_end
        if stepper.n == nstartup && needs_startup_method(method)
            println("n = $(stepper.n): switching to primary ODE method ($method)")
            stepper = change_time_stepper(stepper, method)
            if inplace
                cache = ode_method_cache(method, setup)
            end
        end

        if isadaptive
            if rem(stepper.n, n_adapt_Δt) == 0
                # Change timestep based on operators
                Δt = get_timestep(stepper, cfl; bc_vectors)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, t_end - stepper.t)
        end

        # Perform a single time step with the time integration method
        if inplace
            stepper = step!(stepper, Δt; cache, momentum_cache, bc_vectors)
        else
            stepper = step(stepper, Δt; bc_vectors)
        end

        # Process iteration results with each processor
        for ps ∈ processors
            # Only update each `nupdate`-th iteration
            stepper.n % ps.nupdate == 0 && process!(ps, stepper)
        end
    end

    foreach(finalize!, processors)

    (; V, p) = stepper
    V, p
end
