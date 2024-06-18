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

function lesdatagen(dnsobs, Φ, les, compression, psolver)
    Φu = zero.(Φ(dnsobs[].u, les, compression))
    p = zero(Φu[1])
    div = zero(p)
    ΦF = zero.(Φu)
    FΦ = zero.(Φu)
    c = zero.(Φu)
    results = (; u = fill(Array.(dnsobs[].u), 0), c = fill(Array.(dnsobs[].u), 0))
    temp = nothing
    on(dnsobs) do (; u, F, t)
        Φ(Φu, u, les, compression)
        apply_bc_u!(Φu, t, les)
        Φ(ΦF, F, les, compression)
        momentum!(FΦ, Φu, temp, t, les)
        apply_bc_u!(FΦ, t, les; dudt = true)
        project!(FΦ, les; psolver, div, p)
        for α = 1:length(u)
            c[α] .= ΦF[α] .- FΦ[α]
        end
        push!(results.u, Array.(Φu))
        push!(results.c, Array.(c))
    end
    results
end

"""
    filtersaver(dns, les, filters, compression, psolver_dns, psolver_les; nupdate = 1)

Save filtered DNS data.
"""
filtersaver(dns, les, filters, compression, psolver_dns, psolver_les; nupdate = 1) =
    processor(
        (results, state) -> (; results..., comptime = time() - results.comptime),
    ) do state
        comptime = time()
        (; x) = dns.grid
        T = eltype(x[1])
        F = zero.(state[].u)
        div = zero(state[].u[1])
        p = zero(state[].u[1])
        dnsobs = Observable((; state[].u, F, state[].t))
        data = [
            lesdatagen(dnsobs, Φ, les[i], compression[i], psolver_les[i]) for
            i = 1:length(les), Φ in filters
        ]
        results = (; data, t = zeros(T, 0), comptime)
        temp = nothing
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            momentum!(F, u, temp, t, dns)
            apply_bc_u!(F, t, dns; dudt = true)
            project!(F, dns; psolver = psolver_dns, div, p)
            push!(results.t, t)
            dnsobs[] = (; u, F, t)
        end
        state[] = state[] # Save initial conditions
        results
    end

"""
    create_les_data(
        D = 2,
        Re = 2e3,
        lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
        nles = [ntuple(α -> 64, D)],
        ndns = ntuple(α -> 256, D),
        filters = (FaceAverage(),),
        tburn = typeof(Re)(0.1),
        tsim = typeof(Re)(0.1),
        Δt = typeof(Re)(1e-4),
        create_psolver = psolver_spectral,
        savefreq = 1,
        ArrayType = Array,
        icfunc = (setup, psolver) -> random_field(setup, typeof(Re)(0); psolver),
        rng,
        kwargs...,
    )

Create filtered DNS data.
"""
function create_les_data(;
    D = 2,
    Re = 2e3,
    lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
    nles = [ntuple(α -> 64, D)],
    ndns = ntuple(α -> 256, D),
    filters = (FaceAverage(),),
    tburn = typeof(Re)(0.1),
    tsim = typeof(Re)(0.1),
    Δt = typeof(Re)(1e-4),
    create_psolver = psolver_spectral,
    savefreq = 1,
    ArrayType = Array,
    icfunc = (setup, psolver, rng) -> random_field(setup, typeof(Re)(0); psolver, rng),
    rng,
    kwargs...,
)
    T = typeof(Re)

    compression = [ndns[1] ÷ nles[1] for nles in nles]
    for (c, n) in zip(compression, nles), α = 1:D
        @assert c * n[α] == ndns[α]
    end

    # Build setup and assemble operators
    dns = Setup(
        ntuple(α -> LinRange(lims[α]..., ndns[α] + 1), D)...;
        Re,
        ArrayType,
        kwargs...,
    )
    les = [
        Setup(
            ntuple(α -> LinRange(lims[α]..., nles[α] + 1), D)...;
            Re,
            ArrayType,
            kwargs...,
        ) for nles in nles
    ]

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)
    psolver_les = create_psolver.(les)

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt

    # datasize = Base.summarysize(filtered) / 1e6
    datasize =
        length(filters) *
        (nt ÷ savefreq + 1) *
        sum(prod.(nles)) *
        D *
        2 *
        length(bitstring(zero(T))) / 8 / 1e6
    @info "Generating $datasize Mb of filtered DNS data"

    # Initial conditions
    ustart = icfunc(dns, psolver, rng)

    any(u -> any(isnan, u), ustart) && @warn "Initial conditions contain NaNs"

    # Random body force
    # force_dns =
    #     gaussian_force(xdns...) +
    #     gaussian_force(xdns...) +
    #     # gaussian_force(xdns...) +
    #     # gaussian_force(xdns...) +
    #     gaussian_force(xdns...)
    # force_dns = zero.(u₀)
    # force_les = face_average(force_dns, les, compression)

    _dns = dns
    _les = les
    # _dns = (; dns..., bodyforce = force_dns)
    # _les = (; les..., bodyforce = force_les)

    # Solve burn-in DNS
    (; u, t), outputs =
        solve_unsteady(; setup = _dns, ustart, tlims = (T(0), tburn), Δt, psolver)

    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        setup = _dns,
        ustart = u,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            f = filtersaver(
                _dns,
                _les,
                filters,
                compression,
                psolver,
                psolver_les;
                nupdate = savefreq,
            ),
            # plot = realtimeplotter(; setup = dns, nupdate = 10),
            log = timelogger(; nupdate = 10),
        ),
        psolver,
    )

    # Store result for current IC
    outputs.f
end

"""
    create_io_arrays(data, setups)

Create ``(\\bar{u}, c)`` pairs for training.
"""
function create_io_arrays(data, setups)
    nsample = length(data)
    ngrid, nfilter = size(data[1].data)
    nt = length(data[1].t) - 1
    T = eltype(data[1].t)
    map(CartesianIndices((ngrid, nfilter))) do I
        ig, ifil = I.I
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, (N .- 2)..., D, nt + 1, nsample)
        c = zeros(T, (N .- 2)..., D, nt + 1, nsample)
        ifield = ntuple(Returns(:), D)
        for is = 1:nsample, it = 1:nt+1, α = 1:D
            copyto!(
                view(u, ifield..., α, it, is),
                view(data[is].data[ig, ifil].u[it][α], Iu[α]),
            )
            copyto!(
                view(c, ifield..., α, it, is),
                view(data[is].data[ig, ifil].c[it][α], Iu[α]),
            )
        end
        (; u = reshape(u, (N .- 2)..., D, :), c = reshape(c, (N .- 2)..., D, :))
    end
end
