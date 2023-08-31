_filter_saver(
    dns,
    les,
    KV,
    Kp;
    nupdate = 1,
    bc_vectors_dns = get_bc_vectors(dns, zero(eltype(KV))),
    bc_vectors_les = get_bc_vectors(les, zero(eltype(KV))),
) = processor(
    function (state)
        (; Ω, x) = dns.grid
        Ωbar = les.grid.Ω
        T = eltype(x)
        KVmom = Diagonal(Ωbar) * (KV * Diagonal(1 ./ Ω))
        _t = fill(zero(T), 0)
        _V = fill(zeros(T, 0), 0)
        _p = fill(zeros(T, 0), 0)
        _F = fill(zeros(T, 0), 0)
        _FG = fill(zeros(T, 0), 0)
        _cF = fill(zeros(T, 0), 0)
        _cFG = fill(zeros(T, 0), 0)
        on(state) do (; V, p, t)
            Vbar = KV * V
            pbar = Kp * p
            F, = momentum(V, V, p, t, dns; bc_vectors = bc_vectors_dns, nopressure = true)
            FG, = momentum(
                V,
                V,
                p,
                t,
                dns;
                bc_vectors = bc_vectors_dns,
                nopressure = false,
            )
            Fbar = KVmom * F
            FGbar = KVmom * FG
            FVbar, = momentum(
                Vbar,
                Vbar,
                pbar,
                t,
                les;
                bc_vectors = bc_vectors_les,
                nopressure = true,
            )
            FGVbar, = momentum(
                Vbar,
                Vbar,
                pbar,
                t,
                les;
                bc_vectors = bc_vectors_les,
                nopressure = false,
            )
            cF = Fbar - FVbar
            cFG = FGbar - FGVbar
            push!(_t, t)
            push!(_V, Array(Vbar))
            push!(_p, Array(pbar))
            push!(_F, Array(Fbar))
            push!(_FG, Array(FGbar))
            push!(_cF, Array(cF))
            push!(_cFG, Array(cFG))
        end
        state[] = state[]
        (; t = _t, V = _V, p = _p, F = _F, FG = _FG, cF = _cF, cFG = _cFG)
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
    Δt = tsim / nt

    # Filtered quantities to store
    filtered = (;
        V = zeros(T, nles * nles * 2, nt + 1, nsim),
        p = zeros(T, nles * nles, nt + 1, nsim),
        F = zeros(T, nles * nles * 2, nt + 1, nsim),
        FG = zeros(T, nles * nles * 2, nt + 1, nsim),
        cF = zeros(T, nles * nles * 2, nt + 1, nsim),
        cFG = zeros(T, nles * nles * 2, nt + 1, nsim),
    )

    @info "Generating $(Base.summarysize(filtered) / 1e6) Mb of LES data"

    # Iteration processors
    processors = (
        _filter_saver(
            device(dns),
            device(les),
            device(KV),
            device(Kp);
            bc_vectors_dns = device(get_bc_vectors(dns, T(0))),
            bc_vectors_les = device(get_bc_vectors(les, T(0))),
        ),
        step_logger(; nupdate = 10),
    )

    for isim = 1:nsim
        @info "Generating data for simulation $isim of $nsim"

        # Initial conditions
        V₀, p₀ = random_field(dns; A = T(10_000_000), σ = T(30), s = 5, pressure_solver)

        # Solve burn-in DNS
        @info "Burn-in for simulation $isim of $nsim"
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
        @info "Solving DNS for simulation $isim of $nsim"
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
        filtered.p[:, :, isim] = stack(f.p)
        filtered.F[:, :, isim] = stack(f.F)
        filtered.FG[:, :, isim] = stack(f.FG)
        filtered.cF[:, :, isim] = stack(f.cF)
        filtered.cFG[:, :, isim] = stack(f.cFG)
    end

    filtered
end
