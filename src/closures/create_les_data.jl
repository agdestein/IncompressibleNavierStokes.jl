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
        KVmom = Diagonal(Ωbar) * (KV * Diagonal(1 ./ Ω))
        T = eltype(setup.grid.x)
        _V = fill(zeros(T, 0), 0)
        _F = fill(zeros(T, 0), 0)
        _FG = fill(zeros(T, 0), 0)
        _p = fill(zeros(T, 0), 0)
        _t = fill(zero(T), 0)
        on(step_observer) do (; V, p, t)
            F, = momentum(V, V, p, t, setup; bc_vectors, nopressure = true)
            FG, = momentum(V, V, p, t, setup; bc_vectors, nopressure = false)
            push!(_V, Array(KV * V))
            push!(_F, Array(KVmom * F))
            push!(_FG, Array(KVmom * FG))
            push!(_p, Array(Kp * p))
            push!(_t, t)
        end
        (; V = _V, F = _F, FG = _FG, p = _p, t = _t)
    end;
    nupdate,
)

function create_les_data(
    T;
    viscosity_model = LaminarModel(; Re = T(2_000)),
    lims = (T(0), T(1)),
    nles = 64,
    compression = 4,
    nsim = 10,
    tburn = T(0.1),
    tsim = T(0.1),
    Δt = T(1e-4),
    device = identity,
)
    ndns = compression * nles
    xdns = LinRange(lims..., ndns + 1)
    ydns = LinRange(lims..., ndns + 1)
    xles = xdns[1:compression:end]
    yles = ydns[1:compression:end]

    # Build setup and assemble operators
    dns = Setup(xdns, ydns; viscosity_model)
    les = Setup(xles, yles; viscosity_model)

    # Filter
    (; KV, Kp) = operator_filter(dns.grid, dns.boundary_conditions, compression)

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    pressure_solver = SpectralPressureSolver(dns)

    # Number of time steps to save
    nt = round(Int, tsim / Δt)

    # Filtered quantities to store
    filtered = (;
        V = zeros(T, nles * nles * 2, nt + 1, nsim),
        F = zeros(T, nles * nles * 2, nt + 1, nsim),
        FG = zeros(T, nles * nles * 2, nt + 1, nsim),
        p = zeros(T, nles * nles, nt + 1, nsim),
    )

    @info "Generating $(Base.summarysize(filtered) / 1e6) Mb of LES data"

    # Iteration processors
    processors = (
        _filter_saver(
            device(dns),
            device(KV),
            device(Kp),
            device(les.grid.Ω);
            bc_vectors = device(get_bc_vectors(dns, T(0))),
        ),
        step_logger(; nupdate = 10),
    )

    for isim = 1:nsim
        @info "Generating data for simulation $isim of $nsim"

        # Initial conditions
        V₀, p₀ = random_field(dns; A = T(10_000_000), σ = T(30), s = 5, pressure_solver)

        # Solve burn-in DNS
        @info "Burn-in for IC $isim of $nsim"
        V, p, outputs = solve_unsteady(
            dns,
            V₀,
            p₀,
            (T(0), tburn);
            Δt,
            processors = (step_logger(; nupdate = 10),),
            pressure_solver,
            inplace = true,
            device,
        )

        # Solve DNS and store filtered quantities
        @info "Solving DNS for IC $isim of $nsim"
        V, p, outputs = solve_unsteady(
            dns,
            V,
            p,
            (T(0), tsim);
            Δt,
            processors,
            pressure_solver,
            inplace = true,
            device,
        )
        f = outputs[1]

        # Store result for current IC
        filtered.V[:, :, isim] = stack(f.V)
        filtered.F[:, :, isim] = stack(f.F)
        filtered.FG[:, :, isim] = stack(f.FG)
        filtered.p[:, :, isim] = stack(f.p)
    end

    filtered
end
