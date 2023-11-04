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

_filter_saver(dns, les, comp; nupdate = 1) = processor(function (state)
    (; dimension, x) = dns.grid
    D = dimension()
    F = zero.(state[].u)
    # pbar = zero(volume_average(state[].p, les, comp))
    ubar = zero.(face_average(state[].u, les, comp))
    Fbar = zero.(ubar)
    Fubar = zero.(ubar)
    cF = zero.(ubar)
    _t = fill(zero(eltype(x[1])), 0)
    _u = fill(Array.(ubar), 0)
    # _p = fill(Array.(ubar), 0)
    _F = fill(Array.(ubar), 0)
    # _FG = fill(Array.(ubar), 0)
    _cF = fill(Array.(ubar), 0)
    # _cFG = fill(Array.(ubar), 0)
    on(state) do (; u, p, t)
        face_average!(ubar, u, les, comp)
        apply_bc_u!(ubar, t, les)
        # pbar = Kp * p
        momentum!(F, u, t, dns)
        # G = pressuregradient(p, dns)
        # FG = F .+ G
        face_average!(Fbar, F, les, comp)
        # FGbar = Ku .* FG
        momentum!(Fubar, ubar, t, les)
        # Gubar = pressuregradient(pbar, les)
        # FGubar = Fubar + Gubar
        for α = 1:D
            cF[α] .= Fbar[α] .- Fubar[α]
        end
        # cFG = FGbar .- FGVbar
        push!(_t, t)
        push!(_u, Array.(ubar))
        # push!(_p, Array(pbar))
        # push!(_F, Array.(Fubar))
        # push!(_FG, Array.(FGbar))
        push!(_cF, Array.(cF))
        # push!(_cFG, Array.(cFG))
    end
    state[] = state[] # Save initial conditions
    (;
        t = _t,
        u = _u,
        # p = _p,
        # F = _F,
        # FG = _FG,
        cF = _cF,
        # cFG = _cFG,
    )
end; nupdate)

"""
    create_les_data(
    T;
    D = 2,
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

Create filtered DNS data.
"""
function create_les_data(
    T;
    D = 2,
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
    ndns = compression * nles
    xdns = ntuple(α -> LinRange(lims..., ndns + 1), D)
    xles = ntuple(α -> LinRange(lims..., nles + 1), D)

    # Build setup and assemble operators
    dns = Setup(xdns...; Re, ArrayType)
    les = Setup(xles...; Re, ArrayType)

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    pressure_solver = SpectralPressureSolver(dns)

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt

    # Filtered quantities to store
    (; N) = les.grid
    filtered = (;
        Δt,
        u = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
        # p = fill(fill(zeros(T, N...), 0), 0),
        # F = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
        # FG = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
        cF = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
        # cFG = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
        # force = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
    )

    # @info "Generating $(Base.summarysize(filtered) / 1e6) Mb of LES data"
    @info "Generating $(nsim * (nt + 1) * nles * 3 * 2 * length(bitstring(zero(T))) / 8 / 1e6) Mb of LES data"

    for isim = 1:nsim
        # @info "Generating data for simulation $isim of $nsim"

        # Initial conditions
        u₀, p₀ = random_field(dns, T(0); pressure_solver)

        # Random body force
        # force_dns =
        #     gaussian_force(xdns...) +
        #     gaussian_force(xdns...) +
        #     # gaussian_force(xdns...) +
        #     # gaussian_force(xdns...) +
        #     gaussian_force(xdns...)
        force_dns = zero.(u₀)
        force_les = face_average(force_dns, les, compression)

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
            # processors = (step_logger(; nupdate = 10),),
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
                _filter_saver(_dns, _les, compression),
                # step_logger(; nupdate = 10),
            ),
            pressure_solver,
        )
        f = outputs[1]

        # Store result for current IC
        push!(filtered.u, f.u)
        # push!(filtered.p, f.p)
        # push!(filtered.F, f.F)
        # push!(filtered.FG, f.FG)
        push!(filtered.cF, f.cF)
        # push!(filtered.cFG, f.cFG)
        # push!(filtered.force, f.force)
    end

    filtered
end

"""
    create_io_arrays(data, setup)

Create ``(\\bar{u}, c)`` pairs for training.
"""
function create_io_arrays(data, setup)
    nsample = length(data.u)
    nt = length(data.u[1]) - 1
    D = setup.grid.dimension()
    T = eltype(data.u[1][1][1])
    (; N) = setup.grid
    u = zeros(T, (N .- 2)..., D, nt + 1, nsample)
    c = zeros(T, (N .- 2)..., D, nt + 1, nsample)
    ifield = ntuple(Returns(:), D)
    for i = 1:nsample, j = 1:nt+1, α = 1:D
        copyto!(view(u, ifield..., α, j, i), view(data.u[i][j][α], setup.grid.Iu[α]))
        copyto!(view(c, ifield..., α, j, i), view(data.cF[i][j][α], setup.grid.Iu[α]))
    end
    reshape(u, (N .- 2)..., D, :), reshape(c, (N .- 2)..., D, :)
end
