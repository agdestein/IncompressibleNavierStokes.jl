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
        docopy = true,
        processors = (;),
    )

Solve unsteady problem using `method`.

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl` .

The `processors` are called after every time step.

Note that the `state` observable passed to the `processor.initialize` function
contains vector living on the device, and you may have to move them back to
the host using `Array(u)` and `Array(p)` in the processor.

Return `(; u, p, t), outputs`, where `outputs` is a  named tuple with the
outputs of `processors` with the same field names.
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
    docopy = true,
    processors = (;),
)
    if docopy
        u₀ = copy.(u₀)
        p₀ = copy(p₀)
    end

    t_start, t_end = tlims
    isadaptive = isnothing(Δt)

    # Cache arrays for intermediate computations
    cache = ode_method_cache(method, setup, u₀, p₀)

    # Time stepper
    stepper = create_stepper(method; setup, pressure_solver, u = u₀, p = p₀, t = t_start)

    # Initialize processors for iteration results  
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    if isadaptive
        while stepper.t < t_end
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                Δt = get_timestep(stepper, cfl)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, t_end - stepper.t)

            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    else
        nstep = round(Int, (t_end - t_start) / Δt)
        Δt = (t_end - t_start) / nstep
        for it = 1:nstep
            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    end

    # Final state
    (; u, p, t) = stepper

    # Processor outputs
    outputs = (;
        (k => processors[k].finalize(initialized[k], state) for k in keys(processors))...
    )

    # Return state and outputs
    (; u, p, t), outputs
end

function get_state(stepper)
    (; u, p, t, n) = stepper
    (; u, p, t, n)
end
