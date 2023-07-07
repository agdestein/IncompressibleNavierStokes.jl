"""
    function solve_unsteady(
        setup, V₀, p₀, tlims;
        method = RK44(; T = eltype(V₀)),
        pressure_solver = DirectPressureSolver(setup),
        Δt = nothing,
        cfl = 1,
        n_adapt_Δt = 1,
        inplace = false,
        processors = (),
        device = identity,
    )

Solve unsteady problem using `method`.

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl` .

Each `processor` is called after every `processor.nupdate` time step.

All arrays and operators are passed through the `device` function.
This allows for performing computations on a different device than the host (CPU).
To compute on an Nvidia GPU using CUDA, change

```
solve_unsteady(setup, V₀, p₀, tlims; kwargs...)
```

to the following:

```
using CUDA
solve_unsteady(
    setup, V₀, p₀, tlims;
    device = cu,
    kwargs...
)
```

Note that the `state` observable passed to the `processor.initialize` function
contains vector living on the device, and you may have to move them back to
the host using `Array(V)` and `Array(p)` in the processor.
"""
function solve_unsteady(
    setup,
    V₀,
    p₀,
    tlims;
    method = RK44(; T = eltype(V₀)),
    pressure_solver = DirectPressureSolver(setup),
    Δt = zero(eltype(V₀)),
    cfl = 1,
    n_adapt_Δt = 1,
    inplace = false,
    processors = (),
    device = identity,
)
    t_start, t_end = tlims
    isadaptive = isnothing(Δt)
    if !isadaptive
        nstep = round(Int, (t_end - t_start) / Δt)
        Δt = (t_end - t_start) / nstep
    end

    # Initialize BC arrays (currently only done on host, due to Kronecker
    # products)
    bc_vectors = get_bc_vectors(setup, t_start)

    # Move vectors and operators to device (if any).
    setup = device(setup)
    V₀ = device(V₀)
    p₀ = device(p₀)
    bc_vectors = device(bc_vectors)
    pressure_solver = device(pressure_solver)

    if inplace
        cache = ode_method_cache(method, setup, V₀, p₀)
        momentum_cache = MomentumCache(setup, V₀, p₀)
    end

    # Time stepper
    stepper = create_stepper(
        method;
        setup,
        pressure_solver,
        bc_vectors,
        V = copy(V₀),
        p = copy(p₀),
        t = t_start,
    )

    # Get initial time step
    isadaptive && (Δt = get_timestep(stepper, cfl))

    # Initialize processors for iteration results  
    state = get_state(stepper)
    states = map(ps -> Observable(state), processors)
    initialized = map((ps, o) -> ps.initialize(o), processors, states)

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
            stepper = step!(method, stepper, Δt; cache, momentum_cache)
        else
            stepper = step(method, stepper, Δt)
        end

        # Process iteration results with each processor
        for (ps, o) ∈ zip(processors, states)
            # Only update each `nupdate`-th iteration
            stepper.n % ps.nupdate == 0 && (o[] = get_state(stepper))
        end
    end

    (; V, p, t, n) = stepper
    finalized = map((ps, i) -> ps.finalize(i, get_state(stepper)), processors, initialized)

    (; V, p) = stepper
    V, p, finalized
end

function get_state(stepper)
    (; V, p, t, n) = stepper
    (; V, p, t, n)
end
