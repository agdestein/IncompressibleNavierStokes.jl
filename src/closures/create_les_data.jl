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

_filter_saver(dns, les, comp, pressure_solver; nupdate = 1) =
    processor() do state
        (; dimension, x) = dns.grid
        T = eltype(x[1])
        D = dimension()
        F = zero.(state[].u)
        G = zero.(state[].u)
        Φu = zero.(face_average(state[].u, les, comp))
        q = zero(pressure_additional_solve(pressure_solver, Φu, state[].t, les))
        M = zero(q)
        ΦF = zero.(Φu)
        FΦ = zero.(Φu)
        GΦ = zero.(Φu)
        c = zero.(Φu)
        _t = fill(zero(eltype(x[1])), 0)
        _u = fill(Array.(Φu), 0)
        _c = fill(Array.(Φu), 0)
        on(state) do (; u, p, t)
            momentum!(F, u, t, dns)
            pressuregradient!(G, p, dns)
            for α = 1:D
                F[α] .-= G[α]
            end
            face_average!(Φu, u, les, comp)
            apply_bc_u!(Φu, t, les)
            face_average!(ΦF, F, les, comp)
            momentum!(FΦ, Φu, t, les)
            apply_bc_u!(FΦ, t, les; dudt = true)
            divergence!(M, FΦ, les)
            @. M *= les.grid.Ω

            pressure_poisson!(pressure_solver, q, M)
            apply_bc_p!(q, t, les)
            pressuregradient!(GΦ, q, les)
            for α = 1:D
                FΦ[α] .-= GΦ[α]
                c[α] .= ΦF[α] .- FΦ[α]
            end
            push!(_t, t)
            push!(_u, Array.(Φu))
            push!(_c, Array.(c))
        end
        state[] = state[] # Save initial conditions
        (; t = _t, u = _u, c = _c)
    end

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
    ic_params = (;),
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
    pressure_solver_les = SpectralPressureSolver(les)

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt

    # Filtered quantities to store
    (; N) = les.grid
    filtered = (;
        Δt,
        u = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
        c = fill(fill(ntuple(α -> zeros(T, N...), D), 0), 0),
    )

    # @info "Generating $(Base.summarysize(filtered) / 1e6) Mb of LES data"
    @info "Generating $(nsim * (nt + 1) * nles * 3 * 2 * length(bitstring(zero(T))) / 8 / 1e6) Mb of LES data"

    for isim = 1:nsim
        # @info "Generating data for simulation $isim of $nsim"

        # Initial conditions
        u₀, p₀ = random_field(dns, T(0); pressure_solver, ic_params...)

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
            # processors = (timelogger(; nupdate = 10),),
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
                _filter_saver(_dns, _les, compression, pressure_solver_les),
                # step_logger(; nupdate = 10),
            ),
            pressure_solver,
        )
        f = outputs[1]

        # Store result for current IC
        push!(filtered.u, f.u)
        push!(filtered.c, f.c)
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
        copyto!(view(c, ifield..., α, j, i), view(data.c[i][j][α], setup.grid.Iu[α]))
    end
    reshape(u, (N .- 2)..., D, :), reshape(c, (N .- 2)..., D, :)
end
