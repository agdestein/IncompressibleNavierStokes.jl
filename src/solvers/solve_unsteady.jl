"""
    solve_unsteady(;
        setup,
        tlims,
        ustart,
        tempstart = nothing,
        method = RKMethods.RK44(; T = eltype(u₀[1])),
        psolver = default_psolver(setup),
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
the host using `Array(u)` in the processor.

Return `(; u, t), outputs`, where `outputs` is a  named tuple with the
outputs of `processors` with the same field names.
"""
function solve_unsteady(;
    setup,
    tlims,
    ustart,
    tempstart = nothing,
    method = RKMethods.RK44(; T = eltype(ustart[1])),
    psolver = default_psolver(setup),
    Δt = zero(eltype(ustart[1])),
    cfl = 1,
    n_adapt_Δt = 1,
    docopy = true,
    processors = (;),
    θ = nothing,
)
    docopy && (ustart = copy.(ustart))
    docopy && !isnothing(tempstart) && (tempstart = copy(tempstart))

    tstart, tend = tlims
    isadaptive = isnothing(Δt)

    # Cache arrays for intermediate computations
    cache = ode_method_cache(method, setup, ustart, tempstart)

    # Time stepper
    stepper =
        create_stepper(method; setup, psolver, u = ustart, temp = tempstart, t = tstart)

    # Initialize processors for iteration results  
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    if isadaptive
        while stepper.t < tend
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                Δt = get_timestep(stepper, cfl)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, tend - stepper.t)

            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; θ, cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    else
        nstep = round(Int, (tend - tstart) / Δt)
        Δt = (tend - tstart) / nstep
        for it = 1:nstep
            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; θ, cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    end

    # Final state
    (; u, temp, t) = stepper

    # Processor outputs
    outputs = (;
        (k => processors[k].finalize(initialized[k], state) for k in keys(processors))...
    )

    # Return state and outputs
    (; u, temp, t), outputs
end

function get_state(stepper)
    (; u, temp, t, n) = stepper
    (; u, temp, t, n)
end
