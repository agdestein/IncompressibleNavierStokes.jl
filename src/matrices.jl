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
    i = ilin[Ip][:]
    j = 1:np
    v = ones(eltype(x[1]), np)
    sparse(i, j, v, n, np)
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
    i = zeros(Int, 0)
    for α = 1:D
        I = Iu[α]
        append!(i, ilin[I, α][:])
    end
    j = 1:nu
    v = ones(eltype(x[1]), nu)
    sparse(i, j, v, n, nu)
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
    i = zeros(Int, 0)
    j = zeros(Int, 0)
    v = zeros(eltype(x[1]), 0)

    # Add entries from each of the D velocity components
    I = Ip # These are the indices looped over in the original kernel
    for α = 1:D # Velocity components
        # Original kernel:
        #
        # div[I]  += (u[I, α] - u[I-e(α), α]) / Δ[α][I[α]]
        #
        # becomes
        #
        # div[I] += u[I, α] / Δ[α][I[α]]
        # div[I] += -u[I-e(α), α] / Δ[α][I[α]]
        #
        # We want i, j, v such that div_i = v_ij * u_j
        ΔI = map(I -> Δ[α][I[α]], I)
        append!(i, ilin_p[I][:])
        append!(i, ilin_p[I][:])
        append!(j, ilin_u[I, α][:])
        append!(j, ilin_u[I .- e(α), α][:])
        append!(v, @. 1 / ΔI)
        append!(v, @. -1 / ΔI)
    end

    # Assemble matrix
    sparse(i, j, v, n, n * D)
end

"Pressure gradient matrix."
function pressuregradient_mat(setup)
    (; dimension, N, Iu, Δu, x) = setup
    Δu = adapt(Array, Δu) # Do assembly on CPU
    D = dimension()
    e = Offset(D)
    n = prod(N)
    ilin_u = reshape(1:(n*D), N..., D)
    ilin_p = reshape(1:n, N...)

    # Initialize sparse matrix parts
    i = zeros(Int, 0)
    j = zeros(Int, 0)
    v = zeros(eltype(x[1]), 0)

    # Add entries for each of the D velocity components
    for α = 1:D
        I = Iu[α] # These are the indices looped over in the original kernel

        # Original kernel:
        #
        # G[I, α] = (p[I+e(α)] - p[I]) / Δu[α][I[α]]
        #
        # becomes
        #
        # G[I, α] += p[I+e(α)] / Δu[α][I[α]]
        # G[I, α] -= p[I] / Δu[α][I[α]]
        ΔI = map(I -> Δu[α][I[α]], I)
        append!(i, ilin_u[I, α][:])
        append!(i, ilin_u[I, α][:])
        append!(j, ilin_p[I .+ e(α)][:])
        append!(j, ilin_p[I][:])
        append!(v, @. 1 / ΔI)
        append!(v, @. -1 / ΔI)
    end

    # Assemble matrix
    sparse(i, j, v, n * D, n)
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
    # sum of Dβ * Dβ (different versions of Dβ depending on staggered points)
    (; dimension, N, Iu, Δ, Δu, x) = setup
    Δ = adapt(Array, Δ) # Do assembly on CPU
    Δu = adapt(Array, Δu)
    D = dimension()
    T = eltype(x[1])
    n = prod(N) * D
    ilin = reshape(1:n, N..., D)

    # Initialize sparse matrix parts
    i = zeros(Int, 0) # Output indices
    j = zeros(Int, 0) # Input indices
    v = zeros(T, 0) # Values v_ij

    for α = 1:D # Velocity components
        I = Iu[α] # These are the indices looped over in the original kernel
        for β = 1:D # Differentiation directions
            Δuαβ = map(I -> α == β ? Δu[β][I[β]] : Δ[β][I[β]], I)
            Δa = map(I -> β == α ? Δ[β][I[β]] : Δu[β][I[β]-1], I)
            Δb = map(I -> β == α ? Δ[β][I[β]+1] : Δu[β][I[β]], I)
            eβ = Offset(D)(β)

            # Original
            #
            # ∂a = (u[I, α] - u[I-e(β), α]) / Δa
            # ∂b = (u[I+e(β), α] - u[I, α]) / Δb
            # F[I, α] += (∂b - ∂a) / Δuαβ
            #
            # becomes
            #
            # F[I, α] += u[I - e(β), α] / Δa / Δuαβ
            # F[I, α] += u[I + e(β), α] / Δb / Δuαβ
            # F[I, α] -= u[I, α] * (1 / Δa + 1 / Δb) / Δuαβ
            #
            # We want i, j, v such that F_i = v_ij * u_j
            # They are defined below:

            append!(i, ilin[I, α][:])
            append!(i, ilin[I, α][:])
            append!(i, ilin[I, α][:])
            append!(j, ilin[I .- eβ, α][:])
            append!(j, ilin[I .+ eβ, α][:])
            append!(j, ilin[I, α][:])

            # For some Neumann BC, Δa or Δb are zero (eps),
            # and (right - left) / Δa blows up even if right = left according to BC.
            # Here we manually set the first derivative entry to zero if Δa or Δb are too small.
            a = @. ifelse(Δa > 2 * eps(T), 1 / Δa / Δuαβ, zero(T))
            b = @. ifelse(Δb > 2 * eps(T), 1 / Δb / Δuαβ, zero(T))
            append!(v, a)
            append!(v, b)
            append!(v, @. -(a + b))
        end
    end

    # Assemble matrix
    sparse(i, j, v, n, n)
end
