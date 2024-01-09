"""
    face_average!(v, u, setup, comp)

Average fine grid `u` over coarse volume face. Put result in `v`.
"""
function face_average!(v, u, setup_les, comp)
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

face_average(u, setup_les, comp) = face_average!(
    ntuple(α -> similar(u[1], setup_les.grid.N), length(u)),
    u,
    setup_les,
    comp,
)

"""
    volume_average!(v, u, setup, comp)

Average fine grid `u` over coarse volume. Put result in `v`.
"""
function volume_average!(v, u, setup_les, comp)
    (; grid, workgroupsize) = setup_les
    (; Nu, Iu) = grid
    D = length(u)
    δ = Offset{D}()
    @kernel function Φ!(v, u, ::Val{α}, volume, I0) where {α}
        I = @index(Global, Cartesian)
        J = I0 + comp * (I - oneunit(I))
        s = zero(eltype(v[α]))
        for i in volume
            s += u[α][J+i]
        end
        v[α][I0+I] = s / comp^D
    end
    for α = 1:D
        ndrange = Nu[α]
        I0 = first(Iu[α])
        I0 -= oneunit(I0)
        volume = CartesianIndices(ntuple(β -> 1:comp, D))
        Φ!(get_backend(v[1]), workgroupsize)(v, u, Val(α), volume, I0; ndrange)
    end
    v
end

volume_average(u, setup_les, comp) = volume_average!(
    ntuple(α -> similar(u[1], setup_les.grid.N), length(u)),
    u,
    setup_les,
    comp,
)
