"Get matrix for the Laplacian operator."
function laplacian_mat(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Ip, Δ, Δu) = grid
    backend = get_backend(x[1])
    T = eltype(x[1])
    D = dimension()
    e = Offset(D)
    Ia = first(Ip)
    Ib = last(Ip)
    I = similar(x[1], CartesianIndex{D}, 0)
    J = similar(x[1], CartesianIndex{D}, 0)
    val = similar(x[1], 0)
    I0 = Ia - oneunit(Ia)
    Ω = scalewithvolume!(fill!(scalarfield(setup), 1), setup)
    for α = 1:D
        a, b = boundary_conditions[α]
        i = Ip[ntuple(β -> α == β ? (2:Np[α]-1) : (:), D)...][:]
        ia = Ip[ntuple(β -> α == β ? (1:1) : (:), D)...][:]
        ib = Ip[ntuple(β -> α == β ? (Np[α]:Np[α]) : (:), D)...][:]
        for (aa, bb, j) in [(a, nothing, ia), (nothing, nothing, i), (nothing, b, ib)]
            vala = @.(Ω[j] / Δ[α][getindex.(j, α)] / Δu[α][getindex.(j, α)-1])
            if isnothing(aa)
                J = [J; j .- [e(α)]; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa PressureBC
                J = [J; j]
                I = [I; j]
                val = [val; -vala]
            elseif aa isa PeriodicBC
                J = [J; ib; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa SymmetricBC
                J = [J; ia; j]
                I = [I; j; j]
                val = [val; vala; -vala]
            elseif aa isa DirichletBC
            end

            valb = @.(Ω[j] / Δ[α][getindex.(j, α)] / Δu[α][getindex.(j, α)])
            if isnothing(bb)
                J = [J; j; j .+ [e(α)]]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa PressureBC
                # The weight of the "right" BC is zero, but still needs a J inside Ip, so
                # just set it to ib
                J = [J; j]
                I = [I; j]
                val = [val; -valb]
            elseif bb isa PeriodicBC
                J = [J; j; ia]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa SymmetricBC
                J = [J; j; ib]
                I = [I; j; j]
                val = [val; -valb; valb]
            elseif bb isa DirichletBC
            end
            # val = vcat(
            #     val,
            #     map(I -> Ω[I] / Δ[α][I[α]] / Δu[α][I[α]-1], j),
            #     map(I -> -Ω[I] / Δ[α][I[α]] * (1 / Δu[α][I[α]] + 1 / Δu[α][I[α]-1]), j),
            #     map(I -> Ω[I] / Δ[α][I[α]] / Δu[α][I[α]], j),
        end
    end
    # Go back to CPU, otherwise get following error:
    # ERROR: CUDA error: an illegal memory access was encountered (code 700, ERROR_ILLEGAL_ADDRESS)
    I = Array(I)
    J = Array(J)
    # I = I .- I0
    # J = J .- I0
    I = I .- [I0]
    J = J .- [I0]
    # linear = copyto!(similar(x[1], Int, Np), collect(LinearIndices(Ip)))
    linear = LinearIndices(Ip)
    I = linear[I]
    J = linear[J]

    # Assemble on CPU, since CUDA overwrites instead of adding
    L = sparse(I, J, Array(val))
    # II = copyto!(similar(x[1], Int, length(I)), I)
    # JJ = copyto!(similar(x[1], Int, length(J)), J)
    # sparse(II, JJ, val)

    L
    # Ω isa CuArray ? cu(L) : L
end

# Apply boundary conditions
function bc_u_mat end
function bc_p_mat end
function bc_temp_mat end

function bc_u_mat(setup)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    B = LinearAlgebra.I
    for β = 1:dimension()
        bc_a, bc_b = boundary_conditions[β]
        a = bc_u_mat(bc_a, setup, β, false)
        b = bc_u_mat(bc_b, setup, β, true)
        B = b * a * B
    end
    B
end

function bc_p_mat(setup)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    B = LinearAlgebra.I
    for β = 1:dimension()
        bc_a, bc_b = boundary_conditions[β]
        a = bc_p_mat(bc_a, setup, β, false)
        b = bc_p_mat(bc_b, setup, β, true)
        B = b * a * B
    end
    B
end

function bc_temp_mat(setup)
    (; grid, boundary_conditions) = setup
    (; dimension) = grid
    B = LinearAlgebra.I
    for β = 1:dimension()
        bc_a, bc_b = boundary_conditions[β]
        a = bc_temp_mat(bc_a, setup, β, false)
        b = bc_temp_mat(bc_b, setup, β, true)
        B = b * a * B
    end
    B
end

function bc_u_mat(::PeriodicBC, setup, β, isright = false)
    isright && return LinearAlgebra.I # We do both in one go for "left"
    (; dimension, N, Ip, x) = setup.grid
    T = eltype(x[1])
    D = dimension()
    n = prod(N) * D
    ilin = reshape(1:n, N..., D)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part
    notboundary = ntuple(d -> d == β ? (2:N[d]-1) : (:), D)
    append!(i, ilin[notboundary..., :][:])
    append!(j, ilin[notboundary..., :][:])

    # Periodic part
    eβ = Offset(D)(β)
    Ia = boundary(β, N, Ip, false)
    Ib = boundary(β, N, Ip, true)
    Ja = Ia .+ eβ
    Jb = Ib .- eβ
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
    (; dimension, N, Ip, x) = setup.grid
    T = eltype(x[1])
    D = dimension()
    ilin = reshape(1:prod(N), N)
    i = zeros(Int, 0)
    j = zeros(Int, 0)

    # Identity part
    notboundary = ntuple(d -> d == β ? (2:N[d]-1) : (:), D)
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
    error("DirichletBC not implemented yet")
end

function bc_p_mat(::DirichletBC, setup, β, isright = false)
    error("DirichletBC not implemented yet")
end

function bc_temp_mat(::DirichletBC, setup, β, isright = false)
    error("DirichletBC not implemented yet")
end

function bc_u_mat(::SymmetricBC, setup, β, isright = false)
    error("SymmetricBC not implemented yet")
end

function bc_p_mat(::SymmetricBC, setup, β, isright = false)
    error("SymmetricBC not implemented yet")
end

function bc_temp_mat(::SymmetricBC, setup, β, isright = false)
    error("SymmetricBC not implemented yet")
end

function bc_u_mat(::PressureBC, setup, β, isright = false)
    error("PressureBC not implemented yet")
end

function bc_p_mat(::PressureBC, setup, β, isright = false)
    error("PressureBC not implemented yet")
end

function bc_temp_mat(::PressureBC, setup, β, isright = false)
    error("PressureBC not implemented yet")
end
