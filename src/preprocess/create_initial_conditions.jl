"""
    create_initial_conditions(
        setup,
        t;
        initial_velocity_u,
        initial_velocity_v,
        [initial_velocity_w,]
        initial_pressure = nothing,
        pressure_solver = DirectPressureSolver(setup),
    )

Create initial vectors at starting time `t`. If `p_initial` is a function instead of
`nothing`, calculate compatible IC for the pressure.
"""
function create_initial_conditions end

# 2D version
function create_initial_conditions(
    setup::Setup{T,2},
    t;
    initial_velocity_u,
    initial_velocity_v,
    initial_pressure = nothing,
    pressure_solver = DirectPressureSolver(setup),
) where {T}
    (; grid, operators) = setup
    (; xu, yu, xv, yv, xpp, ypp, Ω) = grid
    (; G, M) = operators

    # Boundary conditions
    bc_vectors = get_bc_vectors(setup, t)
    (; yM) = bc_vectors

    # Allocate velocity and pressure
    u = zero(xu)[:]
    v = zero(xv)[:]
    p = zero(xpp)[:]

    # Initial velocities
    u .= initial_velocity_u.(xu, yu)[:]
    v .= initial_velocity_v.(xv, yv)[:]
    V = [u[:]; v[:]]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv, umom, vmom, k = compute_conservation(V, t, setup; bc_vectors)

    if maxdiv > 1e-12
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."

        # Make velocity field divergence free
        f = M * V + yM
        Δp = pressure_poisson(pressure_solver, f)
        V .-= 1 ./ Ω .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    if isnothing(initial_pressure)
        p = pressure_additional_solve(pressure_solver, V, p, t, setup)
    else
        p .= initial_pressure.(xpp, ypp)[:]
    end

    V, p
end

# 3D version
function create_initial_conditions(
    setup::Setup{T,3},
    t;
    initial_velocity_u,
    initial_velocity_v,
    initial_velocity_w,
    initial_pressure = nothing,
    pressure_solver = DirectPressureSolver(setup),
) where {T}
    (; grid) = setup
    (; xu, yu, zu, xv, yv, zv, xw, yw, zw, xpp, ypp, zpp, Ω) = grid
    (; G, M) = setup.operators

    # Boundary conditions
    bc_vectors = get_bc_vectors(setup, t)
    (; yM) = bc_vectors

    # Allocate velocity and pressure
    u = zero(xu)[:]
    v = zero(xv)[:]
    w = zero(xw)[:]
    p = zero(xpp)[:]

    # Initial velocities
    u .= initial_velocity_u.(xu, yu, zu)[:]
    v .= initial_velocity_v.(xv, yv, zv)[:]
    w .= initial_velocity_w.(xw, yw, zw)[:]
    V = [u; v; w]

    # Kinetic energy and momentum of initial velocity field
    # Iteration 1 corresponds to t₀ = 0 (for unsteady simulations)
    maxdiv, umom, vmom, wmom, k = compute_conservation(V, t, setup; bc_vectors)

    if maxdiv > 1e-12
        @warn "Initial velocity field not (discretely) divergence free: $maxdiv.\n" *
              "Performing additional projection."

        # Make velocity field divergence free
        f = M * V + yM
        Δp = pressure_poisson(pressure_solver, f, setup)
        V .-= 1 ./ Ω .* (G * Δp)
    end

    # Initial pressure: should in principle NOT be prescribed (will be calculated if p_initial)
    if isnothing(initial_pressure)
        p = pressure_additional_solve(pressure_solver, V, p, t, setup)
    else
        p .= initial_pressure.(xpp, ypp, zpp)[:]
    end

    V, p
end

function create_spectrum_2(K, A, σ, s)
    T = typeof(A)
    a =
        A * [
            1 / sqrt((2π)^2 * 2σ^2) *
            exp(-((i - s)^2 + (j - s)^2) / 2σ^2) *
            exp(-2π * im * rand()) for i = 1:K, j = 1:K
        ]
    [
        a reverse(a; dims = 2)
        reverse(a; dims = 1) reverse(a)
    ]
end

function create_spectrum_3(K, A, σ, s)
    a =
        A * [
            1 / sqrt((2π)^3 * 3σ^2) *
            exp(-((i - s)^2 + (j - s)^2 + (k - s)^2) / 2σ^2) *
            exp(-2π * im * rand()) for i = 1:K, j = 1:K, k = 1:K
        ]
    [
        a reverse(a; dims = 2); reverse(a; dims = 1) reverse(a; dims = (1, 2));;;
        reverse(a; dims = 3) reverse(a; dims = (2, 3)); reverse(a; dims = (1, 3)) reverse(a)
    ]
end

"""
    random_field(
        setup, K;
        A = 1e6, σ = 30, s = 5,
        pressure_solver = DirectPressureSolver(setup),
    )

Create random field.

- `K`: Maximum wavenumber
- `A`: Eddy amplitude
- `σ`: Variance
- `s` Wavenumber offset before energy starts decaying
"""
function random_field end

# 2D version
function random_field(
    setup::Setup{T,2},
    K;
    A = convert(eltype(setup.grid.x), 1e6),
    σ = convert(eltype(setup.grid.x), 30),
    s = 5,
    pressure_solver = DirectPressureSolver(setup),
) where {T}
    (; Ω) = setup.grid
    (; G, M) = setup.operators

    u = real.(ifft(create_spectrum_2(K, A, σ, s)))
    v = real.(ifft(create_spectrum_2(K, A, σ, s)))
    V = [reshape(u, :); reshape(v, :)]
    f = M * V
    p = zero(f)

    # Boundary conditions
    bc_vectors = get_bc_vectors(setup, T(0))
    (; yM) = bc_vectors

    # Make velocity field divergence free
    f = M * V + yM
    Δp = pressure_poisson(pressure_solver, f)
    V .-= 1 ./ Ω .* (G * Δp)
    p = pressure_additional_solve(pressure_solver, V, p, T(0), setup; bc_vectors)

    V, p
end

# 3D version
function random_field(
    setup::Setup{T,3},
    K;
    A = convert(eltype(setup.grid.x), 1e6),
    σ = convert(eltype(setup.grid.x), 30),
    s = 5,
    pressure_solver = DirectPressureSolver(setup),
) where {T}
    (; Ω) = setup.grid
    (; G, M) = setup.operators

    u = real.(ifft(create_spectrum_3(K, A, σ, s)))
    v = real.(ifft(create_spectrum_3(K, A, σ, s)))
    w = real.(ifft(create_spectrum_3(K, A, σ, s)))
    V = [reshape(u, :); reshape(v, :); reshape(w, :)]
    f = M * V
    p = zero(f)

    # Boundary conditions
    bc_vectors = get_bc_vectors(setup, T(0))
    (; yM) = bc_vectors

    # Make velocity field divergence free
    f = M * V + yM
    Δp = pressure_poisson(pressure_solver, f)
    V .-= 1 ./ Ω .* (G * Δp)
    p = pressure_additional_solve(pressure_solver, V, p, T(0), setup; bc_vectors)

    V, p
end
