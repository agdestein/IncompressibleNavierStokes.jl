"""
    save_vtk(setup, V, p, t, filename = "output/solution")

Save velocity and pressure field to a VTK file.

In the case of a 2D setup, the velocity field is saved as a 3D vector with a
z-component of zero, as this seems to be preferred by ParaView.
"""
function save_vtk(setup, V, p, t, filename = "output/solution")
    parts = split(filename, "/")
    path = join(parts[1:end-1], "/")
    isdir(path) || mkpath(path)
    (; grid) = setup
    N = setup.grid.dimension()
    if N == 2
        (; xp, yp) = grid
        coords = (xp, yp)
    elseif N == 3
        (; xp, yp, zp) = grid
        coords = (xp, yp, zp)
    end
    vtk_grid(filename, coords...) do vtk
        vels = get_velocity(setup, V, t)
        if N == 2
            # ParaView prefers 3D vectors. Add zero z-component.
            wp = zero(vels[1])
            vels = (vels..., wp)
        end
        vtk["velocity"] = vels
        vtk["pressure"] = p
    end
end
