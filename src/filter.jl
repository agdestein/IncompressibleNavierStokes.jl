"""
    face_average!(v, u, setup, comp)

Average `u` over volume faces. Put result in `v`.
"""
function face_average!(v, u, setup_les, comp)
    (; grid) = setup_les
    (; Nu, Iu) = grid
    D = length(u)
    δ = Offset{D}()
    @kernel function _face_average!(v, u, ::Val{α}, face, I0) where {α}
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
        _face_average!(get_backend(v[1]), WORKGROUP)(v, u, Val(α), face, I0; ndrange)
    end
    # synchronize(get_backend(u[1]))
    v
end

face_average(u, setup_les, comp) = face_average!(
    ntuple(
        α ->
            KernelAbstractions.zeros(get_backend(u[1]), eltype(u[1]), setup_les.grid.N),
        length(u),
    ),
    u,
    setup_les,
    comp,
)
