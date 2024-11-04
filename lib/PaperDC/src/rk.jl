"Encode projection order for use in [`RKProject`](@ref)."
@enumx ProjectOrder begin
    "Project RHS before applying closure term."
    First = 1

    "Project solution instead of RHS (same as `rk`)."
    Last = 2

    "Project RHS after applying closure term."
    Second = 3
end

"""
    RKProject(rk, order)

Runge-Kutta method with different projection order than the default Runge-Kutta method.
"""
struct RKProject{T,R} <: IncompressibleNavierStokes.AbstractODEMethod{T}
    "Runge-Kutta method, for example `RKMethods.RK44()`."
    rk::R

    "Projection order, see [`ProjectOrder`](@ref)."
    projectorder::ProjectOrder.T

    RKProject(rk, projectorder) = new{eltype(rk.A),typeof(rk)}(rk, projectorder)
end

IncompressibleNavierStokes.ode_method_cache(method::RKProject, setup) =
    IncompressibleNavierStokes.ode_method_cache(method.rk, setup)

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
    (; closure_model) = setup
    (; rk, projectorder) = method
    (; A, b, c) = rk
    (; ustart, ku, p) = cache
    nstage = length(b)
    m = closure_model
    @assert projectorder ∈ instances(ProjectOrder.T) "Unknown projectorder: $projectorder"

    # Update current solution
    tstart = t
    copyto!(ustart, u)

    for i = 1:nstage
        # Compute force at current stage i
        apply_bc_u!(u, t, setup)
        momentum!(ku[i], u, temp, t, setup)

        # Project F first
        if projectorder == ProjectOrder.First
            apply_bc_u!(ku[i], t, setup; dudt = true)
            project!(ku[i], setup; psolver, p)
        end

        # Add closure term
        isnothing(m) || (ku[i] .+= m(u, θ))

        # Project F second
        if projectorder == ProjectOrder.Second
            apply_bc_u!(ku[i], t, setup; dudt = true)
            project!(ku[i], setup; psolver, p)
        end

        # Intermediate time step
        t = tstart + c[i] * Δt

        # Apply stage forces
        u .= ustart
        for j = 1:i
            @. u += Δt * A[i, j] * ku[j]
        end

        # Project stage u directly
        # Make velocity divergence free at time t
        if projectorder == ProjectOrder.Last
            apply_bc_u!(u, t, setup)
            project!(u, setup; psolver, p)
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
    (; closure_model) = setup
    (; rk, projectorder) = method
    (; A, b, c) = rk
    nstage = length(b)
    m = closure_model
    @assert projectorder ∈ instances(ProjectOrder.T) "Unknown projectorder: $projectorder"

    # Update current solution (does not depend on previous step size)
    tstart = t
    ustart = u
    ku = ()

    for i = 1:nstage
        # Compute force at current stage i
        u = IncompressibleNavierStokes.apply_bc_u(u, t, setup)
        F = IncompressibleNavierStokes.momentum(u, temp, t, setup)

        # Project F first
        if projectorder == ProjectOrder.First
            F = IncompressibleNavierStokes.apply_bc_u(F, t, setup; dudt = true)
            F = IncompressibleNavierStokes.project(F, setup; psolver)
        end

        # Add closure term
        isnothing(m) || (F = F + m(u, θ))

        # Project F second
        if projectorder == ProjectOrder.Second
            F = IncompressibleNavierStokes.apply_bc_u(F, t, setup; dudt = true)
            F = IncompressibleNavierStokes.project(F, setup; psolver)
        end

        # Store right-hand side of stage i
        ku = (ku..., F)

        # Intermediate time step
        t = tstart + c[i] * Δt

        # Apply stage forces
        u = ustart
        for j = 1:i
            u = @. u + Δt * A[i, j] * ku[j]
        end

        # Project stage u directly
        # Make velocity divergence free at time t
        if projectorder == ProjectOrder.Last
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
