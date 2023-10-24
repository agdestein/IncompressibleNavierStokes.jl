# Body force
function gaussian_force(
    x,
    y;
    σ = eltype(x)(0.05),
    A = eltype(x)(0.002),
    rng = Random.default_rng(),
)
    T = eltype(x)
    Lx = x[end] - x[1]
    Ly = y[end] - y[1]
    xc = x[1] + rand(rng, T) * Lx
    yc = y[1] + rand(rng, T) * Ly
    σx = σ * Lx
    σy = σ * Ly
    ϕ = T(2π) * rand(rng, T)
    f = sum(
        [
            A * exp(-(x - xc - lx)^2 / 2σx^2 - (y - yc - ly)^2 / 2σy^2) for
            y ∈ y[2:end], x ∈ x[2:end], ly ∈ (-Ly, T(0), Ly), lx ∈ (-Lx, T(0), Lx)
        ];
        dims = (3, 4),
    )[
        :,
        :,
        1,
        1,
    ]
    force = cat(sin(ϕ) * f, cos(ϕ) * f; dims = 3)
    force = reshape(force, :)
    force = force .- sum(force) / length(force)
    force
end

_filter_saver(
    dns,
    les,
    Ku,
    Kp;
    nupdate = 1,
) = processor(
    function (state)
        (; dimension, Ω, x) = dns.grid
        D = dimension()
        Ωbar = les.grid.Ω
        T = eltype(x)
        _t = fill(zero(T), 0)
        _u = fill(ntuple(α -> zeros(T, 0), D), 0)
        _p = fill(zeros(T, 0), 0)
        _F = fill(ntuple(α -> zeros(T, 0), D), 0)
        _FG = fill(ntuple(α -> zeros(T, 0), D), 0)
        _cF = fill(ntuple(α -> zeros(T, 0), D), 0)
        _cFG = fill(ntuple(α -> zeros(T, 0), D), 0)
        on(state) do (; u, p, t)
            ubar = Ku .* u
            pbar = Kp * p
            F = momentum(u, t, dns)
            G = pressuregradient(p, dns)
            FG = F .+ G
            Fbar = Ku .* F
            FGbar = Ku .* FG
            FVbar = momentum(ubar, t, les)
            GVbar = pressuregradient(pbar, les)
            FGVbar = FVbar + GVbar
            cF = Fbar .- FVbar
            cFG = FGbar .- FGVbar
            push!(_t, t)
            push!(_u, Array.(ubar))
            push!(_p, Array(pbar))
            push!(_F, Array.(Fbar))
            push!(_FG, Array.(FGbar))
            push!(_cF, Array.(cF))
            push!(_cFG, Array.(cFG))
        end
        state[] = state[]
        (; t = _t, u = _u, p = _p, F = _F, FG = _FG, cF = _cF, cFG = _cFG)
    end;
    nupdate,
)

function create_les_data(
    T;
    dimension,
    Re = T(2_000),
    lims = (T(0), T(1)),
    nles = 64,
    compression = 4,
    nsim = 10,
    tburn = T(0.1),
    tsim = T(0.1),
    Δt = T(1e-4),
    ArrayType = Array,
)
    D = dimension()
    ndns = compression * nles
    xdns = ntuple(α -> LinRange(lims..., ndns + 1), D)
    xles = map(x -> x[1:compression:end], xdns)

    # Build setup and assemble operators
    dns = Setup(xdns...; Re, ArrayType)
    les = Setup(xles...; Re, ArrayType)

    # Filter
    (; Ku, Kp) = operator_filter(dns.grid, dns.boundary_conditions, compression)

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    pressure_solver = SpectralPressureSolver(dns)

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt

    # Filtered quantities to store
    (; N) = les.grid
    filtered = (;
        u = zeros(T, N..., D, nt + 1, nsim),
        p = zeros(T, N..., nt + 1, nsim),
        F = zeros(T, N..., D, nt + 1, nsim),
        FG = zeros(T, N..., D, nt + 1, nsim),
        cF = zeros(T, N..., D, nt + 1, nsim),
        cFG = zeros(T, N..., D, nt + 1, nsim),
        force = zeros(T, N..., D, nsim),
    )

    @info "Generating $(Base.summarysize(filtered) / 1e6) Mb of LES data"

    for isim = 1:nsim
        @info "Generating data for simulation $isim of $nsim"

        # Initial conditions
        u₀, p₀ = random_field(dns; pressure_solver)

        # Random body force
        force_dns = gaussian_force(xdns...) +
            gaussian_force(xdns...) +
            # gaussian_force(xdns...) +
            # gaussian_force(xdns...) +
            gaussian_force(xdns...)
        force_les = Ku .* force_dns

        _dns = (; dns..., force = force_dns)
        _les = (; les..., force = force_les)

        # Solve burn-in DNS
        @info "Burn-in for simulation $isim of $nsim"
        u, p, outputs = solve_unsteady(
            _dns,
            u₀,
            p₀,
            (T(0), tburn);
            Δt,
            processors = (step_logger(; nupdate = 10),),
            pressure_solver,
        )

        # Solve DNS and store filtered quantities
        @info "Solving DNS for simulation $isim of $nsim"
        u, p, outputs = solve_unsteady(
            _dns,
            u,
            p,
            (T(0), tsim);
            Δt,
            processors = (
                _filter_saver(
                    device(_dns),
                    device(_les),
                    device(Ku),
                    device(Kp);
                    bc_vectors_dns = device(get_bc_vectors(_dns, T(0))),
                    bc_vectors_les = device(get_bc_vectors(_les, T(0))),
                ),
                step_logger(; nupdate = 10),
            ),
            pressure_solver,
            inplace = true,
            device,
        )
        f = outputs[1]

        # Store result for current IC
        filtered.u[ntuple(α -> :, D + 2)..., isim] = stack(stack.(f.u))
        filtered.p[ntuple(α -> :, D + 1)..., isim] = stack(f.p)
        filtered.F[ntuple(α -> :, D + 2)..., isim] = stack(stack.(f.F))
        filtered.FG[ntuple(α -> :, D + 2)..., isim] = stack(stack.(f.FG))
        filtered.cF[ntuple(α -> :, D + 2)..., isim] = stack(stack.(f.cF))
        filtered.cFG[ntuple(α -> :, D + 2)..., isim] = stack(stack.(f.cFG))
        filtered.force[ntuple(α -> :, D + 1)..., isim] = stack.(force_les)
    end

    filtered
end
