"Navier-Stokes momentum forcing (convection + diffusion)."
function navierstokes!(force, state, t, params, setup, cache)
    (; u) = state
    fill!(force.u, 0)
    convectiondiffusion!(force.u, state.u, setup)
end

"Navier-Stokes momentum forcing (convection + diffusion)."
function navierstokes(state, t, params, setup)
    c = convection(state.u, setup)
    d = diffusion(state.u, setup)
    (; u = c + d)
end

"Boussinesq forcing (Navier-Stokes + gravity for `u`, convection-diffusion for `temp`)."
function boussinesq!(force, state, t, params, setup, cache)
    (; temperature) = setup
    (; u, temp) = state
    (; diff) = cache
    fill!(force.u, 0)
    fill!(force.temp, 0)
    convectiondiffusion!(force.u, u, setup)
    gravity!(force.u, temp, setup)
    convection_diffusion_temp!(force.temp, u, temp, setup)
    temperature.dodissipation && dissipation!(force.temp, diff, u, setup)
end

"Boussinesq forcing (Navier-Stokes + gravity for `u`, convection-diffusion for `temp`)."
function boussinesq(state, t, params, setup)
    (; temperature) = setup
    (; u, temp) = state
    d = diffusion(u, setup)
    c = convection(u, setup)
    g = gravity(temp, setup)
    fu = @. c + d + g
    ftemp = convection_diffusion_temp(u, temp, setup)
    temperature.dodissipation && (ftemp += dissipation(u, setup))
    (; u = fu, temp = ftemp)
end

get_cache(::typeof(navierstokes!), setup) = nothing
get_cache(::typeof(boussinesq!), setup) = (; diff = vectorfield(setup))

"""
Solve unsteady problem using `method`.

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl`. If `Δt_min` is given, the adaptive time step never goes below it.

The `processors` are called after every time step.

Note that the `state` observable passed to the `processor.initialize` function
contains vector living on the device, and you may have to move them back to
the host using `Array(u)` in the processor.

Return `(; u, t), outputs`, where `outputs` is a  named tuple with the
outputs of `processors` with the same field names.
"""
function solve_unsteady(;
    setup,
    force! = isnothing(setup.temperature) ? navierstokes! : boussinesq!,
    tlims,
    start,
    docopy = true,
    method = LMWray3(; T = eltype(start.u)),
    psolver = default_psolver(setup),
    Δt = nothing,
    Δt_min = nothing,
    cfl = eltype(start.u)(0.9),
    n_adapt_Δt = 1,
    processors = (;),
    params = nothing,
    # Cache arrays for intermediate computations
    ode_cache = get_cache(method, start, setup),
    force_cache = get_cache(force!, setup),
)
    tstart, tend = tlims
    isadaptive = isnothing(Δt)
    if isadaptive
        cflbuf = scalarfield(setup)
    end

    state = docopy ? deepcopy(start) : start

    # Time stepper
    stepper = create_stepper(method; setup, psolver, state, t = tstart)

    # Initialize processors for iteration results
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    if isadaptive
        while stepper.t < tend
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                Δt = cfl * get_cfl_timestep!(cflbuf, stepper.state, setup)
                Δt = isnothing(Δt_min) ? Δt : max(Δt, Δt_min)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, tend - stepper.t)

            # Perform a single time step with the time integration method
            stepper = timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    else
        nstep = round(Int, (tend - tstart) / Δt)
        Δt = (tend - tstart) / nstep
        for it = 1:nstep
            # Perform a single time step with the time integration method
            stepper = timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    end

    # Processor outputs
    outputs = (;
        (k => processors[k].finalize(initialized[k], state) for k in keys(processors))...
    )

    # Return state and outputs
    (; stepper.state..., stepper.t), outputs
end

"Get state `(; u, temp, t, n)` from stepper."
function get_state(stepper)
    (; state, t, n) = stepper
    (; state..., t, n)
end

"Get proposed maximum time step for convection and diffusion terms."
function get_cfl_timestep!(buf, state, setup)
    (; visc, grid) = setup
    (; dimension, Δ, Δu, Iu) = grid
    D = dimension()
    (; u) = state

    # Initial maximum step size
    Δt = eltype(u)(Inf)

    # Check maximum step size in each dimension
    for (α, uα) in enumerate(eachslice(u; dims = D + 1))
        # Diffusion
        Δαmin = minimum(view(Δu[α], Iu[α].indices[α]))
        Δt_diff = Δαmin^2 / visc / 2D

        # Convection
        Δα = reshape(Δu[α], ntuple(Returns(1), α - 1)..., :)
        @. buf = Δα / abs(uα)
        Δt_conv = minimum(view(buf, Iu[α]))

        # Update time step
        Δt = min(Δt, Δt_diff, Δt_conv)
    end

    Δt
end
