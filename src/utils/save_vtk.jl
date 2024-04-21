"""
    save_vtk(setup, u, filename = "output/solution"; fieldnames = [:velocity], psolver)

Save velocity and pressure field to a VTK file.

In the case of a 2D setup, the velocity field is saved as a 3D vector with a
z-component of zero, as this seems to be preferred by ParaView.
"""
function save_vtk(
    setup,
    u,
    t,
    filename = "output/solution";
    fieldnames = [:velocity],
    psolver = DirectPressureSolver(setup),
)
    parts = split(filename, "/")
    path = join(parts[1:end-1], "/")
    isdir(path) || mkpath(path)
    (; grid) = setup
    (; dimension, xp) = grid
    D = dimension()
    xp = Array.(xp)
    vtk_grid(filename, xp...) do vtk
        up = interpolate_u_p(u, setup)
        ωp = interpolate_ω_p(vorticity(u, setup), setup)
        if D == 2
            # ParaView prefers 3D vectors. Add zero z-component.
            up3 = zero(up[1])
            up = (up..., up3)
            ωp = Array(ωp)
        else
            ωp = Array.(ωp)
        end
        vtk["velocity"] = Array.(up)
        :pressure in fieldnames && (vtk["pressure"] = Array(pressure(u, t, setup; psolver)))
        vtk["vorticity"] = ωp
    end
end
