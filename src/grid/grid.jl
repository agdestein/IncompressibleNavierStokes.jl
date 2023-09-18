"""
    Grid(x, boundary_conditions)

Create nonuniform Cartesian box mesh `x[1]` × ... × `x[d]` with boundary
conditions `boundary_conditions`.
"""
function Grid(x, boundary_conditions)
    # Kill all LinRanges etc.
    x = Array.(x)
    xlims = extrema.(x)

    D = length(x)
    dimension = Dimension(D)

    T = eltype(x[1])

    # Add offset positions for ghost volumes
    # For all BC, there is one ghost volume on each side,
    # but not all of the ``d + 1`` fields have a component inside this ghost
    # volume.
    for d = 1:D
        a, b = boundary_conditions[d]
        ghost_a!(a, x[d])
        ghost_b!(b, x[d])
    end

    # Number of finite volumes in each dimension, including ghost volumes
    N = length.(x) .- 1

    # Number of velocity DOFs in each dimension
    Nu = ntuple(D) do α
        ntuple(D) do β
            na = offset_u(boundary_conditions[β][1], α == β, false)
            nb = offset_u(boundary_conditions[β][2], α == β, true)
            N[β] - na - nb
        end
    end

    # Cartesian index ranges of velocity DOFs
    Iu = ntuple(D) do α
        Iuα = ntuple(D) do β
            na = offset_u(boundary_conditions[β][1], α == β, false)
            nb = offset_u(boundary_conditions[β][2], α == β, true)
            1+na:N[β]-nb
        end
        CartesianIndices(Iuα)
    end

    # Number of p DOFs in each dimension
    Np = ntuple(D) do α
        na = offset_p(boundary_conditions[α][1])
        nb = offset_p(boundary_conditions[α][2])
        N[α] - na - nb
    end

    # Cartesian index range of pressure DOFs
    Ip = CartesianIndices(ntuple(D) do α
        na = offset_p(boundary_conditions[α][1])
        nb = offset_p(boundary_conditions[α][2])
        1+na:N[α]-nb
    end)

    xp = ntuple(d -> (x[d][1:end-1] .+ x[d][2:end]) ./ 2, D)

    # Volume widths
    Δ = ntuple(d -> diff(x[d]), D)
    Δu = ntuple(d -> push!(diff(xp[d]), Δ[d][end] / 2), D)

    # Reference volume sizes
    Ω = KernelAbstractions.ones(get_backend(x[1]), T, N...)
    for d = 1:D
        Ω .*= reshape(Δ[d], ntuple(Returns(1), d - 1)..., :)
    end

    # Velocity volume sizes
    Ωu = ntuple(α -> KernelAbstractions.ones(get_backend(x[1]), T, N), D)
    for α = 1:D, β = 1:D
        Ωu[α] .*= reshape((α == β ? Δu : Δ)[β], ntuple(Returns(1), β - 1)..., :)
    end

    # Vorticity volume sizes
    Ωω = KernelAbstractions.ones(get_backend(x[1]), T, N)
    for α = 1:D
        Ωω .*= reshape(Δu[α], ntuple(Returns(1), α - 1)..., :)
    end

    # Velocity volume mid-sections
    Γu = ntuple(α -> ntuple(β -> KernelAbstractions.ones(get_backend(x[1]), T, N), D), D)
    for α = 1:D, β = 1:D, γ in ((1:β-1)..., (β+1:D)...)
        Γu[α][β] .*=
            reshape(γ == β ? 1 : γ == α ? Δu[γ] : Δ[γ], ntuple(Returns(1), γ - 1)..., :)
    end

    # # Velocity points
    # Xu = ntuple(α -> KernelAbstractions.ones(get_backend(x[1]), T, N)

    # Grid quantities
    (;
        dimension,
        N,
        Nu,
        Np,
        Iu,
        Ip,
        xlims,
        x,
        xp,
        Δ,
        Δu,
        Ω,
        # Ωu,
        # Ωω,
        # Γu,
    )
end
