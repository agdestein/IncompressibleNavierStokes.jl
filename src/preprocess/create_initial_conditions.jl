"""
    create_initial_conditions(
        setup;
        initial_velocity_u,
        initial_velocity_v,
        [initial_velocity_w,]
        initial_pressure = nothing,
        pressure_solver,
    )

Create initial vectors. If `p_initial` is a function instead of `nothing`,
calculate compatible IC for the pressure.
"""
function create_initial_conditions end

# 2D version
function create_initial_conditions(
    setup::Setup{T,2};
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure = nothing,
    pressure_solver,
) where {T}
    (; grid, operators) = setup
    (; xu, yu, xv, yv, xpp, ypp, Ω⁻¹) = grid
    (; G, M) = operators

    t = 0.0

    # Allocate velocity and pressure
    u = zero(xu)
    v = zero(xv)
    p = zero(xpp)

    # Initial velocities
    u .= initial_velocity_u.(xu, yu)
    v .= initial_velocity_v.(xv, yv)
    V = [u[:]; v[:]]

    # Kinetic energy and momentum of initial velocity field
    maxdiv = maximum(abs.(M * V))

    if maxdiv > 1e-12
        # Make velocity field divergence free
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."
        f = M * V
        Δp = pressure_poisson(pressure_solver, f, setup)
        V .-= Ω⁻¹ .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    isnothing(initial_pressure) || (p .= initial_pressure.(xpp, ypp))
    p = p[:]

    F = momentum(V, p, t, setup)
    f = M * (Ω⁻¹ .* F)
    Δp = pressure_poisson(pressure_solver, f)
    p = p + Δp

    V, p
end

# 3D version
function create_initial_conditions(
    setup::Setup{T,3};
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure = nothing,
    pressure_solver,
) where {T}
    (; grid) = setup
    (; xu, yu, zu, xv, yv, zv, xw, yw, zw, xpp, ypp, zpp, Ω⁻¹) = grid
    (; G, M) = setup.operators

    t = 0.0

    # Boundary conditions
    # set_bc_vectors!(setup, t)

    # Allocate velocity and pressure
    u = zero(xu)
    v = zero(xv)
    w = zero(xw)
    p = zero(xpp)

    # Initial velocities
    u .= initial_velocity_u.(xu, yu, zu)
    v .= initial_velocity_v.(xv, yv, zv)
    w .= initial_velocity_w.(xw, yw, zw)
    V = [u[:]; v[:]; w[:]]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv = maximum(abs.(M * V))

    if maxdiv > 1e-12
        # Make velocity field divergence free
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."
        f = M * V
        Δp = pressure_poisson(pressure_solver, f, setup)
        V .-= Ω⁻¹ .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    isnothing(initial_pressure) || (p .= initial_pressure.(xpp, ypp, zpp))
    p = p[:]

    F = momentum(V, p, t, setup)
    f = M * (Ω⁻¹ .* F)
    Δp = pressure_poisson(pressure_solver, f)
    p = p + Δp

    V, p
end
