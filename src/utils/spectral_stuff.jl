function spectral_stuff(setup)
    (; dimension, xp, Ip) = setup.grid
    T = eltype(xp[1])
    D = dimension()
    K = size(Ip) .÷ 2
    kx = ntuple(α -> 0:K[α]-1, D)
    k = fill!(similar(xp[1], length.(kx)), 0)
    for α = 1:D
        kα = reshape(kx[α], ntuple(Returns(1), α - 1)..., :, ntuple(Returns(1), D - α)...)
        k .+= kα .^ 2
    end
    k .= sqrt.(k)
    k = reshape(k, :)

    # Sum or average wavenumbers between k and k+1
    kmax = minimum(K) - 1
    nk = ceil(Int, maximum(k))
    kint = 1:kmax
    # ia = similar(xp[1], Int, 0)
    ia = similar(xp[1], Int, length(k))
    ib = sortperm(k)
    # vals = similar(xp[1], 0)
    vals = similar(xp[1], length(k))
    ksort = k[ib]
    jprev = 2 # Do not include constant mode
    for ki = 1:kmax
        j = findfirst(≥(ki + 1 / 2), ksort)
        isnothing(j) && (j = length(k) + 1)
        # ia = [ia; fill!(similar(ia, j - jprev), ki)]
        ia[jprev:j-1] .= ki
        # val = doaverage ? T(1) / (j - jprev) : T(π) * ((ki + 1)^2 - ki^2) / (j - jprev)
        # val = doaverage ? T(1) / (j - jprev) : T(1)
        val = T(1)
        # vals = [vals; fill!(similar(vals, j - jprev), val)]
        vals[jprev:j-1] .= val
        jprev = j
    end
    ia = ia[2:jprev-1]
    ib = ib[2:jprev-1]
    vals = vals[2:jprev-1]
    A = sparse(ia, ib, vals, kmax, length(k))

    (; K, kmax, kx, k, A)
end
