"Create problem setup (stored in a named tuple)."
Setup(;
    x,
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    backend = CPU(),
    workgroupsize = 64,
    temperature = nothing,
    visc = isnothing(temperature) ? convert(eltype(x[1]), 1e-3) : temperature.α1,
) = (;
    grid = Grid(; x, boundary_conditions, backend),
    boundary_conditions,
    visc,
    backend,
    workgroupsize,
    temperature,
)

"""
Create temperature equation setup (stored in a named tuple).

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
