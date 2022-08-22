"""
    solve(
        problem::UnsteadyProblem, method;
        pressure_solver = DirectPressureSolver(problem.setup),
        Δt = nothing,
        CFL = 1,
        n_adapt_Δt = 1,
        processors = Processor[],
        method_startup = nothing,
        nstartup = 1,
    )

Solve unsteady problem using `method`.

The time step is chosen every `n_adapt_Δt` iteration with CFL-number `CFL` if `Δt` is
`nothing`.

For methods that are not self-starting, `nstartup` startup iterations are performed with
`method_startup`.

Each `processor` is called after every `processor.nupdate` time step.
"""
function solve(
    problem::UnsteadyProblem,
    method;
    pressure_solver = DirectPressureSolver(problem.setup),
    Δt = nothing,
    CFL = 1,
    n_adapt_Δt = 1,
    processors = Processor[],
    method_startup = nothing,
    nstartup = 1,
)
    (; setup, V₀, p₀, tlims) = problem
    
    t_start, t_end = tlims
    isadaptive = isnothing(Δt)

    # For methods that need a velocity field at n-1 the first time step
    # (e.g. AB-CN, oneleg beta) use ERK or IRK
    if needs_startup_method(method)
        println("Starting up with method $(method_startup)")
        method_use = method_startup
    else
        method_use = method
    end

    isadaptive && (Δt = 0)
    stepper = TimeStepper(method_use, setup, pressure_solver, V₀, p₀, t_start, Δt)
    isadaptive && (Δt = get_timestep(stepper, CFL))

    # Initialize BC arrays
    set_bc_vectors!(setup, stepper.t)

    # Processors for iteration results  
    for ps ∈ processors
        initialize!(ps, stepper)
        process!(ps, stepper)
    end

    while stepper.t < t_end
        if stepper.n == nstartup && needs_startup_method(method)
            println("n = $(stepper.n): switching to primary ODE method ($method)")
            stepper = change_time_stepper(stepper, method)
        end

        # Change timestep based on operators
        if isadaptive && rem(stepper.n, n_adapt_Δt) == 0 
            Δt = get_timestep(stepper, CFL)
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

    (; V, p) = stepper
    V, p
end
