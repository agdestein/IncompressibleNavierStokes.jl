"""
    create_top_hat_p([T = Float64], N, M)

`N` fine points and `M` coarse points in each dimension.
"""
create_top_hat_p(N, M) = create_top_hat_p(Float64, N, M)
function create_top_hat_p(T, N, M)
    s = N ÷ M
    @assert s * M == N

    i = 1:M
    j = reshape(1:M, 1, :)
    ij = @. i + M * (j - 1)
    ij = repeat(ij, 1, 1, s, s)

    k = reshape(1:s, 1, 1, :)
    l = reshape(1:s, 1, 1, 1, :)

    ijkl = @. s * (i - 1) + k + s * N * (j - 1) + N * (l - 1)

    z = fill(T(1) / s^2, N^2)

    sparse(ij[:], ijkl[:], z)
end

"""
    create_top_hat_u([T = Float64], N, M)

`N` fine points and `M` coarse points in each dimension.
"""
create_top_hat_u(N, M) = create_top_hat_u(Float64, N, M)
function create_top_hat_u(T, N, M)
    s = N ÷ M
    @assert s * M == N

    i = 1:M
    j = reshape(1:M, 1, :)
    ij = @. i + M * (j - 1)
    ij = repeat(ij, 1, 1, 1, s)

    k = fill(1, 1, 1, 1)
    l = reshape(1:s, 1, 1, 1, :)

    ijkl = @. s * (i - 1) + k + s * N * (j - 1) + N * (l - 1)

    z = fill(T(1) / s, N * M)

    sparse(ij[:], ijkl[:], z, M^2, N^2)
end

"""
create_top_hat_v([T = Float64], N, M)

`N` fine points and `M` coarse points in each dimension.
"""
create_top_hat_v(N, M) = create_top_hat_v(Float64, N, M)
function create_top_hat_v(T, N, M)
    s = N ÷ M
    @assert s * M == N

    i = 1:M
    j = reshape(1:M, 1, :)
    ij = @. i + M * (j - 1)
    ij = repeat(ij, 1, 1, s, 1)

    k = reshape(1:s, 1, 1, :, 1)
    l = fill(1, 1, 1, 1, 1)

    ijkl = @. s * (i - 1) + k + s * N * (j - 1) + N * (l - 1)

    z = fill(T(1) / s, N * M)

    sparse(ij[:], ijkl[:], z, M^2, N^2)
end

"""
    create_top_hat_velocity(N, M)

`N` fine points and `M` coarse points in each dimension.
"""
function create_top_hat_velocity(N, M)
    Wu = create_top_hat_u(N, M)
    Wv = create_top_hat_v(N, M)
    blockdiag(Wu, Wv)
end

"""
    operator_filter(grid, boundary_conditions)

Construct filtering operator.
"""
operator_filter(grid, boundary_conditions, s) =
    operator_filter(grid.dimension, grid, boundary_conditions, s)

# 2D version
function operator_filter(::Dimension{2}, grid, boundary_conditions, s)
    (; Nx, Ny, hx, hy, x) = grid
    T = eltype(x)
    N = Nx
    M = N ÷ s

    @assert s * M == N

    # Requirements
    Δx = first(hx)
    Δy = first(hy)
    @assert all(≈(Δx), hx) && all(≈(Δy), hy) "Filter assumes uniform grid"
    @assert all(
        ==((:periodic, :periodic)),
        (boundary_conditions.u.x, boundary_conditions.v.y),
    ) "Filter assumes periodic boundary conditions"
    @assert Nx == Ny

    Kp = create_top_hat_p(T, N, M)
    Ku = create_top_hat_u(T, N, M)
    Kv = create_top_hat_v(T, N, M)
    KV = blockdiag(Ku, Kv)

    (; KV, Kp)
end

# 3D version
function operator_filter(::Dimension{3}, grid, boundary_conditions)
    error("Not implemented")
end
