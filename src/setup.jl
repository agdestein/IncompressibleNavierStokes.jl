"""
    Setup(
        x...;
        boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
        Re = convert(eltype(x[1]), 1_000),
        viscosity_model = LaminarModel(),
        convection_model = NoRegConvectionModel(),
        bodyforce = nothing,
        closure_model = nothing,
        ArrayType = Array,
    )

Create setup.
"""
Setup(
    x...;
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    Re = convert(eltype(x[1]), 1_000),
    viscosity_model = LaminarModel(),
    convection_model = NoRegConvectionModel(),
    bodyforce = nothing,
    closure_model = nothing,
    ArrayType = Array,
) = (;
    grid = Grid(x, boundary_conditions; ArrayType),
    boundary_conditions,
    Re,
    viscosity_model,
    convection_model,
    bodyforce,
    closure_model,
    ArrayType,
)
