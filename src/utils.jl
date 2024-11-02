"Split random number generator seed into `n` new seeds."
splitseed(seed, n) = rand(Xoshiro(seed), UInt32, n)

"""
Get approximate lower and upper limits of a field `x` based on the mean and standard
deviation (``\\mu \\pm n \\sigma``). If `x` is constant, a margin of `1e-4` is enforced. This is required for contour
plotting functions that require a certain range.
"""
function get_lims(x, n = 1.5)
    T = eltype(x)
    μ = mean(x)
    σ = std(x)
    ≈(μ + σ, μ; rtol = sqrt(eps(T)), atol = sqrt(eps(T))) && (σ = sqrt(sqrt(eps(T))))
    (μ - n * σ, μ + n * σ)
end

"""
    plotgrid(x, y; kwargs...)
    plotgrid(x, y, z)

Plot nonuniform Cartesian grid.
"""
function plotgrid end

plotgrid(x, y; kwargs...) = wireframe(
    x,
    y,
    zeros(eltype(x), length(x), length(y));
    axis = (; aspect = DataAspect(), xlabel = "x", ylabel = "y"),
    kwargs...,
)

function plotgrid(x, y, z)
    nx, ny, nz = length(x), length(y), length(z)
    T = eltype(x)

    # x = repeat(x, 1, ny, nz)
    # y = repeat(reshape(y, 1, :, 1), nx, 1, nz)
    # z = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    # vol = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    # volume(x, y, z, vol)
    fig = Figure()

    ax = Axis3(fig[1, 1])
    wireframe!(ax, x, y, fill(z[1], length(x), length(y)))
    wireframe!(ax, x, y, fill(z[end], length(x), length(y)))
    wireframe!(ax, x, fill(y[1], length(z)), repeat(z, 1, length(x))')
    wireframe!(ax, x, fill(y[end], length(z)), repeat(z, 1, length(x))')
    wireframe!(ax, fill(x[1], length(z)), y, repeat(z, 1, length(y)))
    wireframe!(ax, fill(x[end], length(z)), y, repeat(z, 1, length(y)))
    ax.aspect = :data

    ax = Axis(fig[1, 2]; xlabel = "x", ylabel = "y")
    wireframe!(ax, x, y, zeros(T, length(x), length(y)))
    ax.aspect = DataAspect()

    ax = Axis(fig[2, 1]; xlabel = "y", ylabel = "z")
    wireframe!(ax, y, z, zeros(T, length(y), length(z)))
    ax.aspect = DataAspect()

    ax = Axis(fig[2, 2]; xlabel = "x", ylabel = "z")
    wireframe!(ax, x, z, zeros(T, length(x), length(z)))
    ax.aspect = DataAspect()

    fig
end

"Get utilities to compute energy spectrum."
function spectral_stuff(setup; npoint = 100, a = typeof(setup.Re)(1 + sqrt(5)) / 2)
    (; dimension, xp, Np) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    K = div.(Np, 2)

    k = zeros(T, K)
    if D == 2
        kx = reshape(0:K[1]-1, :)
        ky = reshape(0:K[2]-1, 1, :)
        @. k = sqrt(kx^2 + ky^2)
    elseif D == 3
        kx = reshape(0:K[1]-1, :)
        ky = reshape(0:K[2]-1, 1, :)
        kz = reshape(0:K[3]-1, 1, 1, :)
        @. k = sqrt(kx^2 + ky^2 + kz^2)
    end
    k = reshape(k, :)

    # Sum or average wavenumbers between k and k+1
    kmax = minimum(K) - 1
    isort = sortperm(k)
    ksort = k[isort]

    IntArray = typeof(similar(xp[1], Int, 0))
    inds = IntArray[]

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    # κ = logrange(T(1), T(kmax) - 1, npoint)
    # κ = logrange(T(1), T(kmax) / a, npoint)
    # κ = logrange(a, T(kmax) / a, npoint)
    κ = logrange(T(1), T(kmax), npoint)
    # κ = logrange(T(1), T(sqrt(D) * kmax), npoint)
    κ = sort(unique(round.(Int, κ)))
    npoint = length(κ)

    for i = 1:npoint
        if D == 2
            # Dyadic binning - this gives the k^-3 slope in 2D
            jstart = findfirst(≥(κ[i] / a), ksort)
            jstop = findfirst(≥(κ[i] * a), ksort)
            # tol = T(0.01)
            # jstart = findfirst(≥(κ[i] - tol), ksort)
            # jstop = findfirst(≥(κ[i] + 1 - tol), ksort)
        elseif D == 3
            # Linear binning - this gives the k^-5/3 slope in 3D
            tol = T(0.01)
            jstart = findfirst(≥(κ[i] - tol), ksort)
            jstop = findfirst(≥(κ[i] + 1 - tol), ksort)
            # jstart = findfirst(≥(κ[i] - T(0.5) - tol), ksort)
            # jstop = findfirst(≥(κ[i] + T(0.5) + tol), ksort)
        end
        isnothing(jstop) && (jstop = length(ksort) + 1)
        jstop -= 1
        push!(inds, adapt(IntArray, isort[jstart:jstop]))
    end

    (; inds, κ, K)
end

"Get energy spectrum of velocity field."
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
    κ = logrange(T(1), T(sqrt(D) * kmax), npoint)
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
