"""
    Setup(
        x, y;
        Re = 1000,
        viscosity_model = LaminarModel(),
        convection_model = NoRegConvectionModel(),
        u_bc = (x, y, t) -> 0,
        v_bc = (x, y, t) -> 0,
        dudt_bc = nothing,
        dvdt_bc = nothing,
        bc_type = (;
            u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
            v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
            ν = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        ),
        order4 = false,
        bodyforce_u = (x, y) -> 0,
        bodyforce_v = (x, y) -> 0,
        closure_model = nothing,
    )

Create 2D setup.
"""
function Setup(
    x,
    y;
    Re = convert(eltype(x), 1000),
    viscosity_model = LaminarModel(),
    convection_model = NoRegConvectionModel(),
    u_bc = (x, y, t) -> 0,
    v_bc = (x, y, t) -> 0,
    dudt_bc = nothing,
    dvdt_bc = nothing,
    bc_type = (;
        u = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        v = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
        ν = (; x = (:periodic, :periodic), y = (:periodic, :periodic)),
    ),
    order4 = false,
    bodyforce_u = (x, y) -> 0,
    bodyforce_v = (x, y) -> 0,
    steady_force = true,
    closure_model = V -> zero(V),
)
    boundary_conditions =
        BoundaryConditions(u_bc, v_bc; dudt_bc, dvdt_bc, bc_type, T = eltype(x))
    grid = Grid(x, y; boundary_conditions, order4)
    force = SteadyBodyForce(bodyforce_u, bodyforce_v, grid)
    operators = Operators(grid, boundary_conditions)
    (;
        grid,
        boundary_conditions,
        Re,
        viscosity_model,
        convection_model,
        force,
        closure_model,
        operators,
    )
end

"""
    Setup(
        x, y, z;
        Re = convert(eltype(x), 1000),
        viscosity_model = LaminarModel(),
        convection_model = NoRegConvectionModel(),
        u_bc = (x, y, w, t) -> 0.0,
        v_bc = (x, y, w, t) -> 0.0,
        w_bc = (x, y, w, t) -> 0.0,
        dudt_bc = nothing,
        dvdt_bc = nothing,
        dwdt_bc = nothing,
        bc_type = (;
            u = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
            v = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
            w = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
            ν = (;
                x = (:periodic, :periodic),
                y = (:periodic, :periodic),
                z = (:periodic, :periodic),
            ),
        ),
        order4 = false,
        bodyforce_u = (x, y, z) -> 0,
        bodyforce_v = (x, y, z) -> 0,
        bodyforce_w = (x, y, z) -> 0,
        closure_model = nothing,
    )

Create 3D setup.
"""
function Setup(
    x,
    y,
    z;
    Re = convert(eltype(x), 1000),
    viscosity_model = LaminarModel(),
    convection_model = NoRegConvectionModel(),
    u_bc = (x, y, w, t) -> 0,
    v_bc = (x, y, w, t) -> 0,
    w_bc = (x, y, w, t) -> 0,
    dudt_bc = nothing,
    dvdt_bc = nothing,
    dwdt_bc = nothing,
    bc_type = (;
        u = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
        v = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
        w = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
        ν = (;
            x = (:periodic, :periodic),
            y = (:periodic, :periodic),
            z = (:periodic, :periodic),
        ),
    ),
    order4 = false,
    bodyforce_u = (x, y, z) -> 0,
    bodyforce_v = (x, y, z) -> 0,
    bodyforce_w = (x, y, z) -> 0,
    closure_model = V -> zero(V),
)
    boundary_conditions = BoundaryConditions(
        u_bc,
        v_bc,
        w_bc;
        dudt_bc,
        dvdt_bc,
        dwdt_bc,
        bc_type,
        T = eltype(x),
    )
    grid = Grid(x, y, z; boundary_conditions, order4)
    force = SteadyBodyForce(bodyforce_u, bodyforce_v, bodyforce_w, grid)
    operators = Operators(grid, boundary_conditions)
    (;
        grid,
        boundary_conditions,
        Re,
        viscosity_model,
        convection_model,
        force,
        closure_model,
        operators,
    )
end
