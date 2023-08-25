_filter_saver(
    setup,
    KV,
    Kp,
    Ωbar;
    nupdate = 1,
    bc_vectors = get_bc_vectors(setup, zero(eltype(KV))),
) = processor(
    function (step_observer)
        (; Ω) = setup.grid
        KVmom = Ωbar * KV * Diagonal(1 ./ Ω)
        T = eltype(setup.grid.x)
        _V = fill(zeros(T, 0), 0)
        _F = fill(zeros(T, 0), 0)
        _FG = fill(zeros(T, 0), 0)
        _p = fill(zeros(T, 0), 0)
        _t = fill(zero(T), 0)
        on(step_observer) do (; V, p, t)
            F, = momentum(V, V, p, t, setup; bc_vectors, nopressure = true)
            FG, = momentum(V, V, p, t, setup; bc_vectors, nopressure = false)
            push!(_V, KV * Array(V))
            push!(_F, KVmom * Array(F))
            push!(_FG, KVmom * Array(FG))
            push!(_p, Kp * Array(p))
            push!(_t, t)
        end
        (; V = _V, F = _F, FG = _FG, p = _p, t = _t)
    end;
    nupdate,
)

function create_les_data(
    T;
    viscosity_model = LaminarModel(; T(2_000)),
    lims = (T(0), T(1)),
    n_les = 64,
    compression = 4,
    n_ic = 10,
    t_burn = T(0.1),
    t_sim = T(0.1),
    Δt = T(1e-4),
    device = identity,
)
    n_dns = compression * n_les
    x_dns = LinRange(lims..., n_dns + 1)
    y_dns = LinRange(lims..., n_dns + 1)
    x_les = x_dns[1:compression:end]
    y_les = y_dns[1:compression:end]

    # Build setup and assemble operators
    dns = Setup(x_dns, y_dns; viscosity_model)
    les = Setup(x_les, y_les; viscosity_model)

    # Filter
    (; KV, Kp) = operator_filter(dns.grid, dns.boundary_conditions, compression)

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    pressure_solver = SpectralPressureSolver(dns)

    # Number of time steps to save
    n_t = round(Int, t_sim / Δt)

    # Filtered quantities to store
    filtered = (;
        V = zeros(T, n_les * n_les * 2, n_t + 1, n_ic),
        F = zeros(T, n_les * n_les * 2, n_t + 1, n_ic),
        FG = zeros(T, n_les * n_les * 2, n_t + 1, n_ic),
        p = zeros(T, n_les * n_les, n_t + 1, n_ic),
    )

    @info "Generating $(Base.summarysize(filtered) / 1e6) Mb of LES data"

    # Iteration processors
    processors = (
        _filter_saver(
            device(dns),
            KV,
            Kp,
            les.grid.Ω;
            bc_vectors = device(get_bc_vectors(dns, t_start)),
        ),
        step_logger(; nupdate = 10),
    )

    for i_ic = 1:n_ic
        @info "Generating data for IC $i_ic of $n_ic"

        # Initial conditions
        V₀, p₀ = random_field(dns; A = T(10_000_000), σ = T(30), s = 5, pressure_solver)

        # Solve burn-in DNS
        @info "Burn-in for IC $i_ic of $n_ic"
        V, p, outputs = solve_unsteady(
            dns,
            V₀,
            p₀,
            (t_start, t_burn);
            Δt,
            processors = (step_logger(; nupdate = 10),),
            pressure_solver,
            inplace = true,
            device,
        )

        # Solve DNS and store filtered quantities
        @info "Solving DNS for IC $i_ic of $n_ic"
        V, p, outputs = solve_unsteady(
            dns,
            V,
            p,
            (t_burn, t_end);
            Δt,
            processors,
            pressure_solver,
            inplace = true,
            device,
        )
        f = outputs[1]

        # Store result for current IC
        filtered.V[:, :, i_ic] = stack(f.V)
        filtered.F[:, :, i_ic] = stack(f.F)
        filtered.FG[:, :, i_ic] = stack(f.FG)
        filtered.p[:, :, i_ic] = stack(f.p)
    end

    filtered
end
