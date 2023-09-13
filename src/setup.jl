"""
    Setup(
        x;
        boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
        Re = convert(eltype(x[1]), 1_000),
        viscosity_model = LaminarModel(),
        convection_model = NoRegConvectionModel(),
        bodyforce = nothing,
        closure_model = nothing,
    )

Create setup.
"""
function Setup(
    x;
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    Re = convert(eltype(x[1]), 1_000),
    viscosity_model = LaminarModel(),
    convection_model = NoRegConvectionModel(),
    bodyforce = nothing,
    closure_model = nothing,
)
    grid = Grid(x, boundary_conditions)
    (;
        grid,
        boundary_conditions,
        Re,
        viscosity_model,
        convection_model,
        bodyforce,
        closure_model,
    )
end
