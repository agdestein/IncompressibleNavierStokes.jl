"""
    Setup(
        x...;
        boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
        Re = convert(eltype(x[1]), 1_000),
        bodyforce = nothing,
        issteadybodyforce = true,
        closure_model = nothing,
        ArrayType = Array,
        workgroupsize = 64,
    )

Create setup.
"""
function Setup(
    x...;
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    Re = convert(eltype(x[1]), 1_000),
    bodyforce = nothing,
    issteadybodyforce = true,
    closure_model = nothing,
    projectorder = :last,
    ArrayType = Array,
    workgroupsize = 64,
)
    setup = (;
        grid = Grid(x, boundary_conditions; ArrayType),
        boundary_conditions,
        Re,
        bodyforce,
        issteadybodyforce = false,
        closure_model,
        projectorder,
        ArrayType,
        T = eltype(x[1]),
        workgroupsize,
    )
    if !isnothing(bodyforce) && issteadybodyforce
        (; dimension, x, N) = setup.grid
        T = eltype(x[1])
        F = ntuple(α -> zero(similar(x[1], N)), dimension())
        u = ntuple(α -> zero(similar(x[1], N)), dimension())
        bodyforce = bodyforce!(F, u, T(0), setup)
        setup = (; setup..., issteadybodyforce = true, bodyforce)
    end
    setup
end
