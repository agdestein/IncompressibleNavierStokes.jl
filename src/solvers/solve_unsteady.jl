"""
    solve_unsteady(
        setup,
        u₀,
        p₀,
        tlims;
        method = RK44(; T = eltype(u₀[1])),
        pressure_solver = DirectPressureSolver(setup),
        Δt = zero(eltype(u₀[1])),
        cfl = 1,
        n_adapt_Δt = 1,
        inplace = true,
        docopy = true,
        processors = (;),
    )

Solve unsteady problem using `method`.

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl` .

The `processors` are called after every time step.

Return a named tuple with the outputs of `processors` with the same field names.

Note that the `state` observable passed to the `processor.initialize` function
contains vector living on the device, and you may have to move them back to
the host using `Array(u)` and `Array(p)` in the processor.
"""
function solve_unsteady(
    setup,
    u₀,
    p₀,
    tlims;
    method = RK44(; T = eltype(u₀[1])),
    pressure_solver = DirectPressureSolver(setup),
    Δt = zero(eltype(u₀[1])),
    cfl = 1,
    n_adapt_Δt = 1,
    inplace = true,
    docopy = true,
    processors = (;),
)
    if docopy
        u₀ = copy.(u₀)
        p₀ = copy(p₀)
    end

    t_start, t_end = tlims
    isadaptive = isnothing(Δt)
    if !isadaptive
        nstep = round(Int, (t_end - t_start) / Δt)
        Δt = (t_end - t_start) / nstep
    end

    if inplace
        cache = ode_method_cache(method, setup, u₀, p₀)
    end

    # Time stepper
    stepper = create_stepper(method; setup, pressure_solver, u = u₀, p = p₀, t = t_start)

    # Get initial time step
    isadaptive && (Δt = get_timestep(stepper, cfl))

    # Initialize processors for iteration results  
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    while stepper.t < t_end
        if isadaptive
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                Δt = get_timestep(stepper, cfl)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, t_end - stepper.t)
        end

        # Perform a single time step with the time integration method
        if inplace
            stepper = timestep!(method, stepper, Δt; cache)
        else
            stepper = timestep(method, stepper, Δt)
        end

        # Process iteration results with each processor
        state[] = get_state(stepper)
    end

    finalized = (;
        (k => processors[k].finalize(initialized[k], state) for k in keys(processors))...
    )

    # Final state
    (; u, p) = stepper

    # Move output arrays to host
    u, p, finalized
end

function get_state(stepper)
    (; u, p, t, n) = stepper
    (; u, p, t, n)
end
