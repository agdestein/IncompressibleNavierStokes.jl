"""
    Setup(
        x...;
        boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
        bodyforce = nothing,
        issteadybodyforce = true,
        closure_model = nothing,
        ArrayType = Array,
        workgroupsize = 64,
        temperature = nothing,
        Re = isnothing(temperature) ? convert(eltype(x[1]), 1_000) : 1 / temperature.α1,
    )

Create setup.
"""
function Setup(
    x...;
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    bodyforce = nothing,
    issteadybodyforce = true,
    closure_model = nothing,
    projectorder = :last,
    ArrayType = Array,
    workgroupsize = 64,
    temperature = nothing,
    Re = isnothing(temperature) ? convert(eltype(x[1]), 1_000) : 1 / temperature.α1,
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
        temperature,
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

"""
    temperature_equation(; 
        Pr,
        Ra,
        Ge,
        dodissipation = true,
        boundary_conditions,
        gdir = 2,
    )

Temperature equation setup.

The equation is parameterized by three dimensionless numbers (Prandtl number,
Rayleigh number, and Gebhart number), and requires separate boundary conditions
for the `temperature` field. The `gdir` keyword specifies the direction gravity,
while `non_dim_type` specifies the type of non-dimensionalization.
"""
function temperature_equation(;
    Pr,
    Ra,
    Ge,
    dodissipation = true,
    boundary_conditions,
    gdir = 2,
    nondim_type = 1,
)
    if nondim_type == 1
        # free fall velocity, uref = sqrt(beta*g*Delta T*H)
        α1 = sqrt(Pr / Ra)
        α2 = eltype(Pr)(1)
        α3 = Ge * sqrt(Pr / Ra)
        α4 = 1 / sqrt(Pr * Ra)
    elseif nondim_type == 2
        # uref = kappa/H (based on heat conduction time scale)
        α1 = Pr
        α2 = Pr * Ra
        α3 = Ge / Ra
        α4 = 1
    elseif nondim_type == 3
        # uref = sqrt(c*Delta T)
        α1 = sqrt(Pr * Ge / Ra)
        α2 = Ge
        α3 = sqrt(Pr * Ge / Ra)
        α4 = sqrt(Ge / (Pr * Ra))
    end
    γ = α1 / α3
    (; α1, α2, α3, α4, γ, dodissipation, boundary_conditions, gdir)
end
