"""
    Setup(
        x...;
        boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
        Re = convert(eltype(x[1]), 1_000),
        viscosity_model = LaminarModel(),
        bodyforce = nothing,
        closure_model = nothing,
        ArrayType = Array,
        workgroupsize = 64,
    )

Create setup.
"""
Setup(
    x...;
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    Re = convert(eltype(x[1]), 1_000),
    viscosity_model = LaminarModel(),
    bodyforce = nothing,
    closure_model = nothing,
    ArrayType = Array,
    workgroupsize = 64,
) = (;
    grid = Grid(x, boundary_conditions; ArrayType),
    boundary_conditions,
    Re,
    viscosity_model,
    bodyforce,
    closure_model,
    ArrayType,
    workgroupsize,
)
