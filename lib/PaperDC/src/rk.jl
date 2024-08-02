"""
    RKProject(rk, projectorder)

Runge-Kutta method with different projection order.
The Runge-Kutta method `rk` can be for example `RKMethods.RK44()`.

- `projetorder = :first`: Project RHS before applying closure term.
- `projetorder = :second`: Project RHS after applying closure term.
- `projetorder = :last`: Project solution instead of RHS (same as `rk`).
"""
struct RKProject{T,R} <: IncompressibleNavierStokes.AbstractODEMethod{T}
    rk::R
    projectorder::Symbol
    RKProject(rk, projectorder) = new{eltype(rk.A),typeof(rk)}(rk, projectorder)
end

IncompressibleNavierStokes.ode_method_cache(method::RKProject, setup, u, temp) =
    IncompressibleNavierStokes.ode_method_cache(method.rk, setup, u, temp)

IncompressibleNavierStokes.create_stepper(
    method::RKProject;
    setup,
    psolver,
    u,
    temp,
    t,
    n = 0,
) = IncompressibleNavierStokes.create_stepper(method.rk; setup, psolver, u, temp, t, n)

function IncompressibleNavierStokes.timestep!(
    method::RKProject,
    stepper,
    Δt;
    θ = nothing,
    cache,
)
    (; setup, psolver, u, temp, t, n) = stepper
    (; grid, closure_model) = setup
    (; dimension, Iu) = grid
    (; rk, projectorder) = method
    (; A, b, c) = rk
    (; u₀, ku, div, p) = cache
    D = dimension()
    nstage = length(b)
    m = closure_model
    projectorder ∈ (:first, :second, :last) || error("Unknown projectorder: $projectorder")

    # Update current solution
    t₀ = t
    copyto!.(u₀, u)

    for i = 1:nstage
        # Compute force at current stage i
        apply_bc_u!(u, t, setup)
        momentum!(ku[i], u, temp, t, setup)

        # Project F first
        if projectorder == :first
            apply_bc_u!(ku[i], t, setup; dudt = true)
            project!(ku[i], setup; psolver, div, p)
        end

        # Add closure term
        isnothing(m) || map((k, m) -> k .+= m, ku[i], m(u, θ))

        # Project F second
        if projectorder == :second
            apply_bc_u!(ku[i], t, setup; dudt = true)
            project!(ku[i], setup; psolver, div, p)
        end

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Apply stage forces
        for α = 1:D
            u[α] .= u₀[α]
            for j = 1:i
                @. u[α] += Δt * A[i, j] * ku[j][α]
                # @. u[α][Iu[α]] += Δt * A[i, j] * ku[j][α][Iu[α]]
            end
        end

        # Project stage u directly
        # Make velocity divergence free at time t
        if projectorder == :last
            apply_bc_u!(u, t, setup)
            project!(u, setup; psolver, div, p)
        end
    end

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    apply_bc_u!(u, t, setup)

    IncompressibleNavierStokes.create_stepper(method; setup, psolver, u, temp, t, n = n + 1)
end

function IncompressibleNavierStokes.timestep(method::RKProject, stepper, Δt; θ = nothing)
    (; setup, psolver, u, temp, t, n) = stepper
    (; grid, closure_model) = setup
    (; dimension) = grid
    (; rk, projectorder) = method
    (; A, b, c) = rk
    D = dimension()
    nstage = length(b)
    m = closure_model
    projectorder ∈ (:first, :second, :last) || error("Unknown projectorder: $projectorder")

    # Update current solution (does not depend on previous step size)
    t₀ = t
    u₀ = u
    ku = ()

    for i = 1:nstage
        # Compute force at current stage i
        u = IncompressibleNavierStokes.apply_bc_u(u, t, setup)
        F = IncompressibleNavierStokes.momentum(u, temp, t, setup)

        # Project F first
        if projectorder == :first
            F = IncompressibleNavierStokes.apply_bc_u(F, t, setup; dudt = true)
            F = IncompressibleNavierStokes.project(F, setup; psolver)
        end

        # Add closure term
        isnothing(m) || (F = F .+ m(u, θ))

        # Project F second
        if projectorder == :second
            F = IncompressibleNavierStokes.apply_bc_u(F, t, setup; dudt = true)
            F = IncompressibleNavierStokes.project(F, setup; psolver)
        end

        # Store right-hand side of stage i
        ku = (ku..., F)

        # Intermediate time step
        t = t₀ + c[i] * Δt

        # Apply stage forces
        u = u₀
        for j = 1:i
            u = @. u + Δt * A[i, j] * ku[j]
            # u = tupleadd(u, @.(Δt * A[i, j] * ku[j]))
        end

        # Project stage u directly
        # Make velocity divergence free at time t
        if projectorder == :last
            u = IncompressibleNavierStokes.apply_bc_u(u, t, setup)
            u = IncompressibleNavierStokes.project(u, setup; psolver)
        end
    end

    # This is redundant, but Neumann BC need to have _exact_ copies
    # since we divide by an infinitely thin (eps(T)) volume width in the
    # diffusion term
    u = IncompressibleNavierStokes.apply_bc_u(u, t, setup)

    IncompressibleNavierStokes.create_stepper(method; setup, psolver, u, temp, t, n = n + 1)
end
