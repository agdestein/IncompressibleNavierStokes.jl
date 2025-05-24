"""
Represent an `N`-dimensional space.
Returns `N` when called.

```julia-repl
julia> d = Dimension(3)
Dimension{3}()

julia> d()
3
```
"""
struct Dimension{N} end

Dimension(N) = Dimension{N}()

(::Dimension{N})() where {N} = N

"""
Get size of the largest grid element.
"""
function max_size(grid)
    (; Δ) = grid
    m = maximum.(Δ)
    sqrt(sum(m .^ 2))
end

"""
Create a nonuniform grid of `N + 1` points from `a` to `b` using a cosine
profile, i.e.

```math
x_i = a + \\frac{1}{2} \\left( 1 - \\cos \\left( \\pi \\frac{i}{n} \\right) \\right)
(b - a), \\quad i = 0, \\dots, N
```

See also [`stretched_grid`](@ref).
"""
function cosine_grid(a, b, N)
    T = typeof(a)
    i = T.(0:N)
    @. a + (b - a) * (1 - cospi(i / N)) / 2
end

"""
Create a nonuniform grid of `N + 1` points from `a` to `b` with a stretch
factor of `s`. If `s = 1`, return a uniform spacing from `a` to `b`. Otherwise,
return a vector ``x \\in \\mathbb{R}^{N + 1}`` such that ``x_n = a + \\sum_{i =
1}^n s^{i - 1} h`` for ``n = 0, \\dots , N``. Setting ``x_N = b`` then gives
``h = (b - a) \\frac{1 - s}{1 - s^N}``, resulting in

```math
x_n = a + (b - a) \\frac{1 - s^n}{1 - s^N}, \\quad n = 0, \\dots, N.
```

Note that `stretched_grid(a, b, N, s)[n]` corresponds to ``x_{n - 1}``.

See also [`cosine_grid`](@ref).
"""
function stretched_grid(a, b, N, s = 1)
    s > 0 || error("The stretch factor must be positive")
    if s ≈ 1
        LinRange(a, b, N + 1)
    else
        map(i -> a + (b - a) * (1 - s^i) / (1 - s^N), 0:N)
    end
end

"""
Create a nonuniform grid of `N + 1` points from `a` to `b`, as proposed
by Trias et al. [Trias2007](@cite).
"""
function tanh_grid(a, b, N, γ = typeof(a)(1))
    T = typeof(a)
    x = LinRange{T}(0, 1, N + 1)
    @. a + (b - a) * (1 + tanh(γ * (2 * x - 1)) / tanh(γ)) / 2
end

"""
Create useful quantities for Cartesian box mesh
``x[1] \\times \\dots \\times x[d]` with boundary conditions `boundary_conditions`.
Return a named tuple (`[α]` denotes a tuple index) with the following fields:

- `N[α]`: Number of finite volumes in direction `β`, including ghost volumes
- `Nu[α][β]`: Number of `u[α]` velocity DOFs in direction `β`
- `Np[α]`: Number of pressure DOFs in direction `α`
- `Iu[α]`: Cartesian index range of `u[α]` velocity DOFs
- `Ip`: Cartesian index range of pressure DOFs
- `xlims[α]`: Tuple containing the limits of the physical domain (not grid) in the direction `α`
- `x[α]`: α-coordinates of all volume boundaries, including the left point of the first ghost volume
- `xu[α][β]`: β-coordinates of `u[α]` velocity points
- `xp[α]`: α-coordinates of pressure points
- `Δ[α]`: All volume widths in direction `α`
- `Δu[α]`: Distance between pressure points in direction `α`
- `A[α][β]`: Interpolation weights from α-face centers ``x_I`` to ``x_{I \\pm h_β}``

Note that the memory footprint of the redundant 1D-arrays above is negligible
compared to the memory footprint of the 2D/3D-fields used in the code.
"""
function Setup(; x, boundary_conditions, backend = CPU(), workgroupsize = 64)
    # Kill all LinRanges etc.
    x = collect.(x)
    xlims = extrema.(x)

    D = length(x)
    dimension = Dimension(D)

    T = eltype(x[1])

    # Add offset positions for ghost volumes
    # For all BC, there is one ghost volume on each side,
    # but not all of the ``d + 1`` fields have a component inside this ghost
    # volume.
    for d = 1:D
        a, b = boundary_conditions.u[d]
        padghost!(a, x[d], false)
        padghost!(b, x[d], true)
    end

    # Number of finite volumes in each dimension, including ghost volumes
    N = @. length(x) - 1

    # Number of velocity DOFs in each dimension
    Nu = ntuple(D) do α
        ntuple(D) do β
            na = offset_u(boundary_conditions.u[β][1], false, α == β)
            nb = offset_u(boundary_conditions.u[β][2], true, α == β)
            N[β] - na - nb
        end
    end

    # Number of pressure DOFs in each dimension
    Np = ntuple(D) do α
        na = offset_p(boundary_conditions.u[α][1], false)
        nb = offset_p(boundary_conditions.u[α][2], true)
        N[α] - na - nb
    end

    # Cartesian index ranges of velocity DOFs
    Iu = ntuple(D) do α
        Iuα = ntuple(D) do β
            na = offset_u(boundary_conditions.u[β][1], false, α == β)
            nb = offset_u(boundary_conditions.u[β][2], true, α == β)
            (1+na):(N[β]-nb)
        end
        CartesianIndices(Iuα)
    end

    # Cartesian index range of pressure DOFs
    Ip = CartesianIndices(ntuple(D) do α
        na = offset_p(boundary_conditions.u[α][1], false)
        nb = offset_p(boundary_conditions.u[α][2], true)
        (1+na):(N[α]-nb)
    end)

    # Coordinates of velocity points
    xu = ntuple(D) do α
        ntuple(D) do β
            if α == β
                x[β][2:end]
            else
                (x[β][1:(end-1)] .+ x[β][2:end]) ./ 2
            end
        end
    end

    # Coordinates of pressure points
    xp = map(x -> (x[1:(end-1)] .+ x[2:end]) ./ 2, x)

    # Volume widths
    # Infinitely thin widths are set to `eps(T)` to avoid division by zero
    Δ = map(x) do x
        Δx = diff(x)
        Δx = max.(Δx, eps(eltype(x)))
        Δx
    end

    Δu = ntuple(D) do d
        Δx = push!(diff(xp[d]), Δ[d][end] / 2)
        Δx = max.(Δx, eps(eltype(Δx)))
        Δx
    end

    # Interpolate from face to center
    A_coll = ntuple(D) do j
        Aj1 = fill(T(1 / 2), N[j])
        Aj1[1] = 1
        Aj2 = fill(T(1 / 2), N[j])
        Aj2[end] = 1
        Aj1, Aj2
    end

    # Interpolate from center to face
    A_stag = ntuple(D) do j
        Aj2 = [(x[j][n] - xp[j][n-1]) / Δu[j][n-1] for n = 2:N[j]]
        Aj1 = 1 .- Aj2
        pushfirst!(Aj1, 1)
        push!(Aj2, 1)
        Aj1, Aj2
    end

    # Store quantities
    (;
        e = Offset(D),
        xlims,
        dimension,
        N,
        Nu,
        Np,

        # Keep those as ranges
        Iu,
        Ip,

        # Put arrays on GPU, if requested
        adapt(backend, (; x, xu, xp, Δ, Δu, A_coll, A_stag))...,
        boundary_conditions,
        backend,
        workgroupsize,
    )
end
