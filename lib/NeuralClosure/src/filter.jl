"""
Discrete DNS filter.

Subtypes `ConcreteFilter` should implement the in-place method:

    (::ConcreteFilter)(v, u, setup_les, compression)

which filters the DNS field `u` and put result in LES field `v`.
Then the out-of place method:

    (::ConcreteFilter)(u, setup_les, compression)

automatically becomes available.
"""
abstract type AbstractFilter end

"Average fine grid velocity field over coarse volume face."
struct FaceAverage <: AbstractFilter end

"Average fine grid velocity field over coarse volume."
struct VolumeAverage <: AbstractFilter end

(Φ::AbstractFilter)(u, setup_les, compression) =
    Φ(vectorfield(setup_les), u, setup_les, compression)

function (::FaceAverage)(v, u, setup_les, comp)
    (; grid, backend, workgroupsize) = setup_les
    (; dimension, Nu, Iu) = grid
    D = dimension()
    @kernel function Φ!(v, u, ::Val{α}, face, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v))
        for i in face
            s += u[J+i, α]
        end
        v[I0+I, α] = s / comp^(D - 1)
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = getoffset(Iu[α])
        face = CartesianIndices(ntuple(β -> β == α ? (comp:comp) : (1:comp), D))
        Φ!(backend, workgroupsize)(v, u, Val(α), face, I0; ndrange)
    end
    v
end

"Reconstruct DNS velocity `u` from LES velocity `v`."
function reconstruct!(u, v, setup_dns, setup_les, comp)
    (; grid, boundary_conditions, backend, workgroupsize) = setup_les
    (; dimension, N) = grid
    D = dimension()
    e = Offset(D)
    @assert all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions)
    @kernel function R!(u, v, ::Val{α}, volume) where {α}
        J = @index(Global, Cartesian)
        I = oneunit(J) + comp * J
        J = oneunit(J) + J
        Jleft = J - e(α)
        Jleft.I[α] == 1 && (Jleft += (N[α] - 2) * e(α))
        for i in volume
            s = zero(eltype(v[α]))
            s += (comp - i.I[α]) * v[J, α]
            s += i.I[α] * v[Jleft, α]
            u[I-i, α] = s / comp
        end
    end
    for α = 1:D
        ndrange = N .- 2
        volume = CartesianIndices(ntuple(β -> 0:comp-1, D))
        R!(backend, workgroupsize)(u, v, Val(α), volume; ndrange)
    end
    u
end

"Reconstruct DNS velocity field. See also [`reconstruct!`](@ref)."
reconstruct(v, setup_dns, setup_les, comp) =
    reconstruct!(vectorfield(setup_dns), v, setup_dns, setup_les, comp)

function (::VolumeAverage)(v, u, setup_les, comp)
    (; grid, boundary_conditions, backend, workgroupsize) = setup_les
    (; dimension, N, Nu, Iu) = grid
    D = dimension()
    @assert all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions)
    @kernel function Φ!(v, u, ::Val{α}, volume, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v))
        # n = 0
        for i in volume
            # Periodic extension
            K = J + i
            K = mod1.(K.I, comp .* (N .- 2))
            K = CartesianIndex(K)
            s += u[K, α]
            # n += 1
        end
        n = (iseven(comp) ? comp + 1 : comp) * comp^(D - 1)
        v[I0+I, α] = s / n
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = getoffset(Iu[α])
        volume = CartesianIndices(
            ntuple(
                β ->
                    α == β ?
                    iseven(comp) ? (div(comp, 2):div(comp, 2)+comp) :
                    (div(comp, 2)+1:div(comp, 2)+comp) : (1:comp),
                D,
            ),
        )
        Φ!(backend, workgroupsize)(v, u, Val(α), volume, I0; ndrange)
    end
    v
end
