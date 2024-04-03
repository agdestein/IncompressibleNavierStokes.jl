function spectral_stuff(setup; npoint = 100, a = typeof(setup.Re)(1 + sqrt(5)) / 2)
    (; dimension, xp, Ip) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    K = size(Ip) .÷ 2
    k = zeros(T, K)
    for α = 1:D
        kα =
            reshape(0:K[α]-1, ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)

    # Sum or average wavenumbers between k and k+1
    kmax = minimum(K) - 1
    isort = sortperm(k)
    ksort = k[isort]
    ia = zeros(Int, 0)
    ib = zeros(Int, 0)
    vals = zeros(T, 0)

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    # logκ = LinRange(T(0), log(T(kmax) - 1), npoint)
    logκ = LinRange(T(0), log(T(kmax) / a), npoint)
    # logκ = LinRange(log(a), log(T(kmax) / a), npoint)
    # logκ = LinRange(T(0), log(T(kmax)), npoint)
    κ = exp.(logκ)
    κ = sort(unique(round.(Int, κ)))
    npoint = length(κ)

    for i = 1:npoint
        jstart = findfirst(≥(κ[i] / a), ksort)
        jstop = findfirst(≥(κ[i] * a), ksort)
        # jstart = findfirst(≥(κ[i] - T(1.01)), ksort)
        # jstop = findfirst(≥(κ[i] + T(1.01)), ksort)
        isnothing(jstop) && (jstop = length(ksort) + 1)
        jstop -= 1
        nk = jstop - jstart + 1
        append!(ia, fill(i, nk))
        append!(ib, isort[jstart:jstop])
        append!(vals, fill(T(1), nk))
    end
    IntArray = typeof(similar(xp[1], Int, 0))
    TArray = typeof(similar(xp[1], 0))
    ia = adapt(IntArray, ia)
    ib = adapt(IntArray, ib)
    vals = adapt(TArray, vals)
    A = sparse(ia, ib, vals, npoint, length(k))

    (; A, κ, K)
end

function get_spectrum(setup; npoint = 100, a = typeof(e.setup.Re)(1 + sqrt(5)) / 2)
    (; dimension, xp, Ip) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    @assert all(==(size(Ip, 1)), size(Ip))

    K = size(Ip, 1) .÷ 2
    kmax = K - 1
    k = ntuple(
        i -> reshape(0:kmax, ntuple(Returns(1), i - 1)..., :, ntuple(Returns(1), D - i)...),
        D,
    )

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    logκ = LinRange(T(0), log(T(kmax) / a), npoint)
    κ = exp.(logκ)
    κ = sort(unique(round.(Int, κ)))
    npoint = length(κ)

    masks = map(κ) do κ
        if D == 2
            @. (κ / a)^2 ≤ k[1]^2 + k[2]^2 < (κ * a)^2
        elseif D == 3
            @. (κ / a)^2 ≤ k[1]^2 + k[2]^2 + k[3]^2 < (κ * a)^2
        else
            error("Not implemented")
        end
    end

    BoolArray = typeof(similar(xp[1], Bool, ntuple(Returns(0), D)...))
    masks = adapt.(BoolArray, masks)
    (; κ, masks, K)
end
