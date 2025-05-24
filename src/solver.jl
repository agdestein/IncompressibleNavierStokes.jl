"Navier-Stokes momentum forcing (convection + diffusion)."
function navierstokes!(force, state, t; setup, cache, viscosity)
    (; u) = state
    fill!(force.u, 0)
    convectiondiffusion!(force.u, state.u, setup, viscosity)
end

"Navier-Stokes momentum forcing (convection + diffusion)."
function navierstokes(state, t; setup, viscosity)
    c = convection(state.u, setup)
    d = diffusion(state.u, setup, viscosity)
    (; u = c + d)
end

"Boussinesq forcing (Navier-Stokes + gravity for `u`, convection-diffusion for `temp`)."
function boussinesq!(
    force,
    state,
    t;
    setup,
    cache,
    viscosity,
    conductivity,
    gdir,
    gravity,
    dodissipation,
)
    (; u, temp) = state
    fill!(force.u, 0)
    fill!(force.temp, 0)
    convectiondiffusion!(force.u, u, setup, viscosity)
    applygravity!(force.u, temp, setup, gdir, gravity)
    convection_diffusion_temp!(force.temp, u, temp, setup, conductivity)
    dodissipation && dissipation!(force.temp, u, setup, viscosity)
end

"Boussinesq forcing (Navier-Stokes + gravity for `u`, convection-diffusion for `temp`)."
function boussinesq(state, t; setup, viscosity, conductivity, gdir, gravity, dodissipation)
    (; u, temp) = state
    d = diffusion(u, setup, viscosity)
    c = convection(u, setup)
    g = applygravity(temp, setup, gdir, gravity)
    fu = @. c + d + g
    ftemp = convection_diffusion_temp(u, temp, setup, conductivity)
    dodissipation && (ftemp += dissipation(u, setup, viscosity))
    (; u = fu, temp = ftemp)
end

get_cache(::typeof(navierstokes!), setup) = nothing
get_cache(::typeof(boussinesq!), setup) = nothing

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
    tlims,
    start,
    force! = haskey(start, :temp) ? boussinesq! : navierstokes!,
    docopy = true,
    method = LMWray3(; T = eltype(start.u)),
    psolver = default_psolver(setup),
    Δt = nothing,
    Δt_min = nothing,
    cfl = eltype(start.u)(0.9),
    n_adapt_Δt = 1,
    processors = (;),
    params,
    # Cache arrays for intermediate computations
    ode_cache = get_cache(method, start, setup),
    force_cache = get_cache(force!, setup),
)
    tstart, tend = tlims
    isadaptive = isnothing(Δt)

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
                Δt = cfl * propose_timestep(force!, stepper.state, setup, params)
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

function propose_timestep(::typeof(diffusion!), state, setup, params)
    (; dimension, Δ, Δu, Iu) = setup
    D = dimension()

    # Check maximum step size in each dimension
    minimum(1:D) do α
        Δαmin = minimum(view(Δu[α], Iu[α].indices[α]))
        Δαmin^2 / params.viscosity / 2D
    end
end

broadcastreduce(f, op, args...; kwargs...) =
    reduce(op, Broadcast.instantiate(Broadcast.broadcasted(f, args...); kwargs...))

function propose_timestep(::typeof(convection!), state, setup, params)
    (; dimension, Δ, Δu, Iu) = setup
    D = dimension()
    (; u) = state

    # Check maximum step size in each dimension
    minimum(1:D) do α
        uα = selectdim(u, D + 1, α)
        Δα = view(Δu[α], Iu[α].indices[α])
        Δα = reshape(Δα, ntuple(Returns(1), α - 1)..., :)
        uα = view(uα, Iu[α])
        broadcastreduce(min, Δα, uα) do Δα, uα
            Δα / abs(uα)
        end
    end
end

function propose_timestep(::typeof(convection_diffusion_temp!), state, setup, params)
    (; dimension, Δ, Ip) = setup
    D = dimension()

    # Check maximum step size in each dimension
    minimum(1:D) do α
        Δαmin = minimum(view(Δ[α], Ip.indices[α]))
        Δαmin^2 / params.conductivity / 2D
    end
end

propose_timestep(::typeof(navierstokes), state, setup, params) =
    propose_timestep(navierstokes!, state, setup, params)
propose_timestep(::typeof(navierstokes!), state, setup, params) = min(
    propose_timestep(convection!, state, setup, params),
    propose_timestep(diffusion!, state, setup, params),
)
propose_timestep(::typeof(boussinesq), state, setup, params) =
    propose_timestep(boussinesq!, state, setup, params)
propose_timestep(::typeof(boussinesq!), state, setup, params) = min(
    propose_timestep(navierstokes!, state, setup, params),
    propose_timestep(convection_diffusion_temp!, state, setup, params),
)
