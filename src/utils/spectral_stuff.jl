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
    logκ = LinRange(T(0), log(T(kmax) / a), npoint)
    # logκ = LinRange(log(a), log(T(kmax) / a), npoint)
    # logκ = LinRange(T(0), log(T(kmax)), npoint)
    κ = exp.(logκ)
    κ = sort(unique(round.(Int, κ)))
    npoint = length(κ)

    for i = 1:npoint
        jstart = findfirst(≥(κ[i] / a), ksort)
        jstop = findfirst(≥(κ[i] * a), ksort)
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
