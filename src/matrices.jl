# # Note
#
# This file contains "matrix versions" of linear operators from `operators.jl`
# and `boundary_conditions.jl`. The matrices are used in exactly the same way as
# the matrix-free operators, e.g.
#
#     u = apply_bc_u(u, t, setup) # u is a (D+1)-array
#     d = diffusion(u, setup; use_viscosity = false) # d is a (D+1)-array
#
# becomes
#
#     uvec = bc_u_mat(setup) * u[:] # flatten u to a vector first
#     dvec = diffusion_mat(setup) * uvec # dvec is a vector
#     d = reshape(dvec, size(u)) # Go back to (D+1)-array

"""
Create matrix for padding inner scalar field with boundary volumes.
This can be useful for algorithms that require vectors with degrees of freedom only,
and not the ghost volumes. To go back, simply transpose the matrix.

See also: [`pad_vectorfield_mat`](@ref).
"""
function pad_scalarfield_mat(setup)
    (; N, Np, Ip, x) = setup
    n = prod(N)
    np = prod(Np)
    ilin = reshape(1:n, N)
    a = ilin[Ip][:]
    b = 1:np
    v = ones(eltype(x[1]), np)
    sparse(a, b, v, n, np)
end

"""
Create matrix for padding inner vector field with boundary volumes,
similar to [`pad_scalarfield_mat`](@ref).
"""
function pad_vectorfield_mat(setup)
    (; dimension, N, Nu, Iu, x) = setup
    D = dimension()
    n = prod(N) * D
    nu = sum(prod.(Nu))
    ilin = reshape(1:n, N..., D)
    a = zeros(Int, 0)
    for i = 1:D
        I = Iu[i]
        append!(a, ilin[I, i][:])
    end
    b = 1:nu
    v = ones(eltype(x[1]), nu)
    sparse(a, b, v, n, nu)
end

"""
Create matrix for applying boundary conditions to velocity fields `u`.
This matrix only applies the boundary conditions depending on `u` itself (e.g. [`PeriodicBC`](@ref)).
It does not apply constant boundary conditions (e.g. non-zero [`DirichletBC`](@ref)).
"""
function bc_u_mat end

"Matrix for applying boundary conditions to pressure fields `p`."
function bc_p_mat end

"Matrix for applying boundary conditions to temperature fields `temp`."
function bc_temp_mat end

function bc_u_mat(setup)
    (; dimension, boundary_conditions) = setup
    B = LinearAlgebra.I # BC mat should preserve inputs
    for β = 1:dimension()
        bc_a, bc_b = boundary_conditions.u[β]
        a = bc_u_mat(bc_a, setup, β, false) # Left BC
        b = bc_u_mat(bc_b, setup, β, true) # Right BC
        B = b * a * B # Apply left and right BC for given dimension
    end
    B
end

function bc_p_mat(setup)
    (; dimension, boundary_conditions) = setup
    B = LinearAlgebra.I
    for β = 1:dimension()
        bc_a, bc_b = boundary_conditions.u[β]
        a = bc_p_mat(bc_a, setup, β, false)
        b = bc_p_mat(bc_b, setup, β, true)
        B = b * a * B
    end
    B
end

function bc_temp_mat(setup)
    (; dimension, boundary_conditions) = setup
    B = LinearAlgebra.I
    for β = 1:dimension()
        bc_a, bc_b = boundary_conditions.temp[β]
        a = bc_temp_mat(bc_a, setup, β, false)
        b = bc_temp_mat(bc_b, setup, β, true)
        B = b * a * B
    end
    B
end

function bc_u_mat(::PeriodicBC, setup, β, isright = false)
    isright && return LinearAlgebra.I # We do both in one go for "left"
    (; dimension, N, Ip, x) = setup
    T = eltype(x[1])
    D = dimension()
    n = prod(N) * D # Total number of points in vector fields
    ilin = reshape(1:n, N..., D) # Maps Cartesian to linear indices
    i = zeros(Int, 0) # Output indices
    j = zeros(Int, 0) # Input indices

    # Identity part: Make sure to zero out boundary outputs first, so
    # that corner BC are not added twice since we do this in each dimension
    notboundary = ntuple(d -> d == β ? (2:(N[d]-1)) : (:), D)
    append!(i, ilin[notboundary..., :][:])
    append!(j, ilin[notboundary..., :][:])

    # Periodic part
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Ip, false) # Left boundary
    Ib = boundary(β, N, Ip, true) # Right boundary
    Ja = Ia .+ eβ # Left points that should be copied to right boundary
    Jb = Ib .- eβ # Right points that should be copied to left boundary
    append!(i, ilin[Ia, :][:])
    append!(i, ilin[Ib, :][:])
    append!(j, ilin[Jb, :][:])
    append!(j, ilin[Ja, :][:])

    # For periodic BC, all values are 1
    v = ones(T, length(i))

    sparse(i, j, v, n, n)
end

function bc_p_mat(::PeriodicBC, setup, β, isright = false)
    isright && return LinearAlgebra.I # We do both in one go for "left"
    (; dimension, N, Ip, x) = setup
    T = eltype(x[1])
    D = dimension()
    ilin = reshape(1:prod(N), N)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part
    notboundary = ntuple(d -> d == β ? (2:(N[d]-1)) : (:), D)
    append!(i, ilin[notboundary...][:])
    append!(j, ilin[notboundary...][:])

    # Periodic part
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Ip, false)
    Ib = boundary(β, N, Ip, true)
    Ja = Ia .+ eβ
    Jb = Ib .- eβ
    append!(i, ilin[Ia][:])
    append!(i, ilin[Ib][:])
    append!(j, ilin[Jb][:])
    append!(j, ilin[Ja][:])

    # For periodic BC, all values are 1
    v = ones(T, length(i))

    sparse(i, j, v, prod(N), prod(N))
end

bc_temp_mat(bc::PeriodicBC, setup, β, isright = false) =
    apply_bc_p_mat(bc, setup, β, isright)

function bc_u_mat(::DirichletBC, setup, β, isright = false)
    (; dimension, N, Iu, x) = setup
    T = eltype(x[1])
    D = dimension()
    n = prod(N) * D
    ilin = reshape(1:n, N..., D)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part: Make sure not to include current boundary points,
    # but do include unused ghost volumes
    for α = 1:D
        inside = Iu[α].indices[β]
        inds = if isright
            vcat(1:inside[end], (inside[end]+2):N[β])
        else
            vcat(1:(inside[1]-2), inside[1]:N[β])
        end
        notboundary = ntuple(d -> d == β ? inds : (:), D)
        append!(i, ilin[notboundary..., α][:])
        append!(j, ilin[notboundary..., α][:])
    end

    # The boundary condition values do not depend on `u`, and are thus
    # not part of the matrix.

    # The only values are identity matrix at non-boundary output points
    v = ones(T, length(i))

    # Assemble matrix
    sparse(i, j, v, n, n)
end

bc_p_mat(::DirichletBC, setup, β, isright = false) = LinearAlgebra.I

function bc_temp_mat(::DirichletBC, setup, β, isright = false)
    (; dimension, N, Ip, x) = setup
    T = eltype(x[1])
    D = dimension()
    n = prod(N)
    ilin = reshape(1:n, N...)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part: Make sure not to include current boundary points,
    # but do include unused ghost volumes
    inside = Ip.indices[β]
    inds = if isright
        vcat(1:inside[end], (inside[end]+2):N[β])
    else
        vcat(1:(inside[1]-2), inside[1]:N[β])
    end
    notboundary = ntuple(d -> d == β ? inds : (:), D)
    append!(i, ilin[notboundary...][:])
    append!(j, ilin[notboundary...][:])

    # The boundary condition values do not depend on `temp`, and are thus
    # not part of the matrix.

    # The only values are identity matrix at non-boundary output points
    v = ones(T, length(i))

    # Assemble matrix
    sparse(i, j, v, n, n)
end

function bc_u_mat(::SymmetricBC, setup, β, isright = false)
    (; dimension, N, Nu, Iu, x) = setup
    D = dimension()
    e = Offset(D)
    n = prod(N) * D
    ilin = reshape(1:n, N..., D)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part: Make sure not to include current boundary points, but do
    # include # unused ghost volumes
    for α = 1:D
        # Identity part
        inside = Iu[α].indices[β]
        inds = if isright
            vcat(1:inside[end], (inside[end]+2):N[β])
        else
            vcat(1:(inside[1]-2), inside[1]:N[β])
        end
        notboundary = ntuple(d -> d == β ? inds : (:), D)
        append!(i, ilin[notboundary..., α][:])
        append!(j, ilin[notboundary..., α][:])

        # Symmetric part
        if α != β
            I = boundary(β, N, Iu[α], isright)
            J = isright ? I .- e(β) : I .+ e(β)
            # Kernel: @. u[I, α] = u[J, α]
            append!(i, ilin[I, α][:])
            append!(j, ilin[J, α][:])
        end
    end

    # All values are 1
    v = ones(eltype(x[1]), length(i))

    # Assemble matrix
    sparse(i, j, v, n, n)
end

function bc_p_mat(::SymmetricBC, setup, β, isright = false)
    (; dimension, N, Ip, x) = setup
    T = eltype(x[1])
    D = dimension()
    n = prod(N)
    ilin = reshape(1:n, N...)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part: Make sure not to include current boundary points,
    # but do include unused ghost volumes
    inside = Ip.indices[β]
    inds = if isright
        vcat(1:inside[end], (inside[end]+2):N[β])
    else
        vcat(1:(inside[1]-2), inside[1]:N[β])
    end
    notboundary = ntuple(d -> d == β ? inds : (:), D)
    append!(i, ilin[notboundary...][:])
    append!(j, ilin[notboundary...][:])

    # Symmetric part
    e = Offset(D)
    I = boundary(β, N, Ip, isright)
    J = isright ? I .- e(β) : I .+ e(β)
    # Kernel: @. p[I] = p[J]
    append!(i, ilin[I][:])
    append!(j, ilin[J][:])

    # All values are 1
    v = ones(T, length(i))

    # Assemble matrix
    sparse(i, j, v, n, n)
end

bc_temp_mat(bc::SymmetricBC, setup, β, isright = false) = bc_p_mat(bc, setup, β, isright)

function bc_u_mat(::PressureBC, setup, β, isright = false)
    (; dimension, N, Nu, Iu, x) = setup
    D = dimension()
    e = Offset(D)
    n = prod(N) * D
    ilin = reshape(1:n, N..., D)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part: Make sure not to include current boundary points,
    # but do include unused ghost volumes
    for α = 1:D
        # Identity part
        inside = Iu[α].indices[β]
        inds = if isright
            vcat(1:inside[end], (inside[end]+2):N[β])
        else
            vcat(1:(inside[1]-2), inside[1]:N[β])
        end
        notboundary = ntuple(d -> d == β ? inds : (:), D)
        append!(i, ilin[notboundary..., α][:])
        append!(j, ilin[notboundary..., α][:])

        # Neumann part
        I = boundary(β, N, Iu[α], isright)
        J = isright ? I .- e(β) : I .+ e(β)
        # Kernel: @. u[I, α] = u[J, α]
        append!(i, ilin[I, α][:])
        append!(j, ilin[J, α][:])
    end

    # All values are 1
    v = ones(eltype(x[1]), length(i))

    # Assemble matrix
    sparse(i, j, v, n, n)
end

function bc_p_mat(::PressureBC, setup, β, isright = false)
    (; dimension, N, Ip, x) = setup
    T = eltype(x[1])
    D = dimension()
    n = prod(N)
    ilin = reshape(1:n, N...)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part: Make sure not to include current boundary points,
    # but do include unused ghost volumes
    inside = Ip.indices[β]
    inds = if isright
        vcat(1:inside[end], (inside[end]+2):N[β])
    else
        vcat(1:(inside[1]-2), inside[1]:N[β])
    end
    notboundary = ntuple(d -> d == β ? inds : (:), D)
    append!(i, ilin[notboundary...][:])
    append!(j, ilin[notboundary...][:])

    # The boundary condition values do not depend on `p`, and are thus
    # not part of the matrix.

    # The only values are identity matrix at non-boundary output points
    v = ones(T, length(i))

    # Assemble matrix
    sparse(i, j, v, n, n)
end

bc_temp_mat(bc::PressureBC, setup, β, isright = false) = bc_p_mat(bc, setup, β, isright)

"Divergence matrix."
function divergence_mat(setup)
    (; dimension, N, Ip, Δ, x) = setup
    Δ = adapt(Array, Δ) # Do assembly on CPU
    D = dimension()
    e = Offset(D)
    n = prod(N)
    ilin_p = reshape(1:n, N)
    ilin_u = reshape(1:(n*D), N..., D)

    # Initialize sparse matrix parts
    a = zeros(Int, 0)
    b = zeros(Int, 0)
    v = zeros(eltype(x[1]), 0)

    # These are the indices looped over in the original kernel
    A = CartesianIndex(map(Returns(2), N))
    B = CartesianIndex(map(n -> n - 1, N))
    I = A:B

    # Add entries from each of the D velocity components
    for i = 1:D # Velocity components
        # Original kernel:
        #
        # div[I]  += (u[I, i] - u[I-e(i), i]) / Δ[i][I[i]]
        #
        # becomes
        #
        # div[I] += u[I, i] / Δ[i][I[i]]
        # div[I] += -u[I-e(i), i] / Δ[i][I[i]]
        #
        # We want a, b, v such that div_a = v_ab * u_b
        ΔI = map(I -> Δ[i][I[i]], I)
        append!(a, ilin_p[I][:])
        append!(a, ilin_p[I][:])
        append!(b, ilin_u[I, i][:])
        append!(b, ilin_u[left.(I, i), i][:])
        append!(v, @. 1 / ΔI)
        append!(v, @. -1 / ΔI)
    end

    # Assemble matrix
    sparse(a, b, v, n, n * D)
end

"Pressure gradient matrix."
function pressuregradient_mat(setup)
    (; dimension, N, Δu, x) = setup
    Δu = adapt(Array, Δu) # Do assembly on CPU
    D = dimension()
    n = prod(N)
    ilin_u = reshape(1:(n*D), N..., D)
    ilin_p = reshape(1:n, N...)

    # Initialize sparse matrix parts
    a = zeros(Int, 0)
    b = zeros(Int, 0)
    v = zeros(eltype(x[1]), 0)

    # These are the indices looped over in the original kernel
    A = CartesianIndex(map(Returns(2), N))
    B = CartesianIndex(map(n -> n - 1, N))
    I = A:B

    # Add entries for each of the D velocity components
    for i = 1:D
        # Original kernel:
        #
        # G[I, i] = (p[I+e(i)] - p[I]) / Δu[i][I[i]]
        #
        # becomes
        #
        # G[I, i] += p[I+e(i)] / Δu[i][I[i]]
        # G[I, i] -= p[I] / Δu[i][I[i]]
        ΔI = map(I -> Δu[i][I[i]], I)
        append!(a, ilin_u[I, i][:])
        append!(a, ilin_u[I, i][:])
        append!(b, ilin_p[right.(I, i)][:])
        append!(b, ilin_p[I][:])
        append!(v, @. 1 / ΔI)
        append!(v, @. -1 / ΔI)
    end

    # Assemble matrix
    sparse(a, b, v, n * D, n)
end

"Volume-size matrix."
function volume_mat(setup)
    (; N) = setup
    n = prod(N)
    v = scalewithvolume!(fill!(scalarfield(setup), 1), setup)
    v = adapt(Array, v) # Do assembly on CPU
    sparse(1:n, 1:n, v[:], n, n)
end

"""
Get matrix for the Laplacian operator (for the pressure-Poisson equation).
This matrix takes scalar field inputs restricted to the actual degrees of freedom.
"""
function laplacian_mat(setup)
    P = pad_scalarfield_mat(setup)
    Bp = bc_p_mat(setup)
    Bu = bc_u_mat(setup)
    G = pressuregradient_mat(setup)
    M = divergence_mat(setup)
    Ω = volume_mat(setup)
    P' * Ω * M * Bu * G * Bp * P
end

"Diffusion matrix."
function diffusion_mat(setup)
    # Note: This matrix could also be implemented as
    # sum of Dj * Dj (different versions of Dj depending on staggered points)
    (; dimension, N, Iu, Δ, Δu, x) = setup
    Δ = adapt(Array, Δ) # Do assembly on CPU
    Δu = adapt(Array, Δu)
    D = dimension()
    T = eltype(x[1])
    n = prod(N) * D
    ilin = reshape(1:n, N..., D)

    # Initialize sparse matrix parts
    a = zeros(Int, 0) # Output indices
    b = zeros(Int, 0) # Input indices
    v = zeros(T, 0) # Values v_ab

    # These are the indices looped over in the original kernel
    A = CartesianIndex(map(Returns(2), N))
    B = CartesianIndex(map(n -> n - 1, N))
    I = A:B

    for i = 1:D # Velocity components
        for j = 1:D # Differentiation directions
            Δuαβ = map(I -> i == j ? Δu[j][I[j]] : Δ[j][I[j]], I)
            Δa = map(I -> j == i ? Δ[j][I[j]] : Δu[j][I[j]-1], I)
            Δb = map(I -> j == i ? Δ[j][I[j]+1] : Δu[j][I[j]], I)

            # Original
            #
            # ∂a = (u[I, i] - u[I-e(j), i]) / Δa
            # ∂b = (u[I+e(j), i] - u[I, i]) / Δb
            # F[I, i] += (∂b - ∂a) / Δuαβ
            #
            # becomes
            #
            # F[I, i] += u[I - e(j), i] / Δa / Δuαβ
            # F[I, i] += u[I + e(j), i] / Δb / Δuαβ
            # F[I, i] -= u[I, i] * (1 / Δa + 1 / Δb) / Δuαβ
            #
            # We want a, b, v such that F_a = v_ab * u_b
            # They are defined below:

            append!(a, ilin[I, i][:])
            append!(a, ilin[I, i][:])
            append!(a, ilin[I, i][:])
            append!(b, ilin[left.(I, j), i][:])
            append!(b, ilin[right.(I, j), i][:])
            append!(b, ilin[I, i][:])

            # For some Neumann BC, Δa or Δb are zero (eps),
            # and (right - left) / Δa blows up even if right = left according to BC.
            # Here we manually set the first derivative entry to zero if Δa or Δb are too small.
            aa = @. ifelse(Δa > 2 * eps(T), 1 / Δa / Δuαβ, zero(T))
            bb = @. ifelse(Δb > 2 * eps(T), 1 / Δb / Δuαβ, zero(T))
            append!(v, aa)
            append!(v, bb)
            append!(v, @. -(aa + bb))
        end
    end

    # Assemble matrix
    sparse(a, b, v, n, n)
end
