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
    p = scalarfield(les)
    Φu = vectorfield(les)
    FΦ = vectorfield(les)
    ΦF = vectorfield(les)
    c = vectorfield(les)
    results = (; u = fill(Array.(Φu), 0), c = fill(Array.(c), 0))
    temp = nothing
    on(dnsobs) do (; u, F, t)
        Φ(Φu, u, les, compression)
        apply_bc_u!(Φu, t, les)
        Φ(ΦF, F, les, compression)
        momentum!(FΦ, Φu, temp, t, les)
        apply_bc_u!(FΦ, t, les; dudt = true)
        project!(FΦ, les; psolver, p)
        for α = 1:length(u)
            c[α] .= ΦF[α] .- FΦ[α]
        end
        push!(results.u, Array.(Φu))
        push!(results.c, Array.(c))
    end
    results
end

"""
Save filtered DNS data.
"""
filtersaver(
    dns,
    les,
    filters,
    compression,
    psolver_dns,
    psolver_les;
    nupdate = 1,
    F = vectorfield(dns),
    p = scalarfield(dns),
) =
    processor(
        (results, state) -> (; results..., comptime = time() - results.comptime),
    ) do state
        comptime = time()
        t = fill(state[].t, 0)
        dnsobs = Observable((; state[].u, F, state[].t))
        data = map(
            splat((i, Φ) -> lesdatagen(dnsobs, Φ, les[i], compression[i], psolver_les[i])),
            Iterators.product(eachindex(les), filters),
        )
        # if !isnothing(les.bodyforce)
        #     # Overwrite LES body force field with filtered DNS bodyforce.
        #     # In principle, they should be very similar, but the original LES
        #     # body force field is obtained with pointwise evaluation,
        #     # while the one appearing in the filterd DNS equation is
        #     # discretely filtered.
        #     @assert les.issteadybodyforce
        #     Φ(les.bodyforce, dns.bodyforce, les, compression)
        # end
        results = (;
            data,
            t,
            # les.bodyforce,
            comptime,
        )
        temp = nothing
        on(state) do (; u, t, n)
            n % nupdate == 0 || return
            momentum!(F, u, temp, t, dns)
            apply_bc_u!(F, t, dns; dudt = true)
            project!(F, dns; psolver = psolver_dns, p)
            push!(results.t, t)
            dnsobs[] = (; u, F, t)
        end
        state[] = state[] # Save initial conditions
        results
    end

"""
Create filtered DNS data.
"""
function create_les_data(;
    D,
    Re,
    lims,
    nles,
    ndns,
    filters,
    tburn,
    tsim,
    savefreq,
    Δt = nothing,
    method = RKMethods.RK44(; T = typeof(Re)),
    create_psolver = default_psolver,
    ArrayType = Array,
    icfunc = (setup, psolver, rng) -> random_field(setup, typeof(Re)(0); psolver, rng),
    processors = (; log = timelogger(; nupdate = 10)),
    rng,
    kwargs...,
)
    T = typeof(Re)

    compression = div.(ndns, nles)
    @assert all(==(ndns), compression .* nles)

    # Build setup and assemble operators
    dns = Setup(; x = ntuple(α -> LinRange(lims..., ndns + 1), D), Re, ArrayType, kwargs...)
    les = map(
        nles -> Setup(;
            x = ntuple(α -> LinRange(lims..., nles + 1), D),
            Re,
            ArrayType,
            kwargs...,
        ),
        nles,
    )

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)
    psolver_les = create_psolver.(les)

    # Initial conditions
    ustart = icfunc(dns, psolver, rng)

    any(u -> any(isnan, u), ustart) && @warn "Initial conditions contain NaNs"

    # Define cache outside `solve_unsteady` to re-use the arrays for the
    # filtered DNS force
    cache = IncompressibleNavierStokes.ode_method_cache(method, dns)

    # Solve burn-in DNS
    # The initial spectrum is artificial, but this small simulation will
    # create a more realistic spectrum for the DNS simulation
    (; u, t), outputs = solve_unsteady(;
        setup = dns,
        ustart,
        tlims = (T(0), tburn),
        Δt,
        method,
        psolver,
        cache,
        processors,
    )

    # Solve DNS and store filtered quantities
    # Use the result of the burn-in as initial conditions
    _, outputs = solve_unsteady(;
        setup = dns,
        ustart = u,
        tlims = (T(0), tsim),
        Δt,
        method,
        cache,
        processors = (;
            processors...,
            f = filtersaver(
                dns,
                les,
                filters,
                compression,
                psolver,
                psolver_les;
                nupdate = savefreq,

                # Reuse arrays from cache.
                # Since processors are called outside
                # Runge-Kutta steps, there is no danger
                # in overwriteng the arrays.
                F = cache.u₀,
                p = cache.p,
            ),
        ),
        psolver,
    )

    outputs.f
end

"""
Create ``(\\bar{u}, c)`` pairs for a-priori training.
"""
function create_io_arrays(data, setup)
    (; dimension, N, Iu) = setup.grid
    T = eltype(data[1].t)
    D = dimension()
    colons = ntuple(Returns(:), D)
    fields = map((:u, :c)) do usym
        u = map(data) do trajectory
            nt = length(trajectory.t)
            u = zeros(T, (N .- 2)..., D, nt)
            for it = 1:nt, α = 1:D
                copyto!(
                    view(u, colons..., α, it),
                    view(getfield(trajectory, usym)[it][α], Iu[α]),
                )
            end
            u
        end
        u = cat(u...; dims = D + 2)
        usym => u
    end
    (; fields...)
end
