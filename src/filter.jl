abstract type AbstractFilter end

struct FaceAverage <: AbstractFilter end
struct VolumeAverage <: AbstractFilter end

(Φ::AbstractFilter)(u, setup_les, compression) =
    Φ(ntuple(α -> similar(u[1], setup_les.grid.N), length(u)), u, setup_les, compression)

"""
    (::FaceAverage)(v, u, setup_les)

Average fine grid `u` over coarse volume face. Put result in `v`.
"""
function (::FaceAverage)(v, u, setup_les, comp)
    (; grid, workgroupsize) = setup_les
    (; Nu, Iu) = grid
    D = length(u)
    δ = Offset{D}()
    @kernel function Φ!(v, u, ::Val{α}, face, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v[α]))
        for i in face
            s += u[α][J+i]
        end
        v[α][I0+I] = s / comp^(D - 1)
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        face = CartesianIndices(ntuple(β -> β == α ? (comp:comp) : (1:comp), D))
        Φ!(get_backend(v[1]), workgroupsize)(v, u, Val(α), face, I0; ndrange)
    end
    v
end

"""
    (::VolumeAverage)(v, u, setup_les, comp)

Average fine grid `u` over coarse volume. Put result in `v`.
"""
function (::VolumeAverage)(v, u, setup_les, comp)
    (; grid, boundary_conditions, workgroupsize) = setup_les
    (; N, Nu, Iu) = grid
    D = length(u)
    δ = Offset{D}()
    @assert all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions)
    @kernel function Φ!(v, u, ::Val{α}, volume, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v[α]))
        for i in volume
            # Periodic extension
            K = J + i
            K = mod1.(K.I, comp .* (N .- 2))
            K = CartesianIndex(K)
            s += u[α][K]
        end
        n = (iseven(comp) ? comp : comp + 1) * comp^(D - 1)
        v[α][I0+I] = s / n
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        volume = CartesianIndices(
            ntuple(
                β ->
                    α == β ? iseven(comp) ? (comp÷2:comp÷2+comp) : (comp+1:2comp+1) :
                    (1:comp),
                D,
            ),
        )
        Φ!(get_backend(v[1]), workgroupsize)(v, u, Val(α), volume, I0; ndrange)
    end
    v
end
