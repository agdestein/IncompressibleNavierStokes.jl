"""
    save_vtk(V, p, filename = "output/solution")

Save velocity and pressure field to a VTK file.
"""
function save_vtk(V, p, t, setup, filename = "output/solution")
    (; xp, yp, zp) = setup.grid
    N = get_dimension(setup.grid)
    if N == 2
        coords = (xp, yp)
    elseif N == 3
        coords = (xp, yp, zp)
    end
    vtk_grid(filename, coords...) do vtk
        vels = get_velocity(V, t, setup)
        if N == 2
            # ParaView prefers 3D vectors. Add zero z-component.
            wp = zeros(size(vels[1]))
            vels = (vels..., wp)
        end
        vtk["velocity"] = vels
        vtk["pressure"] = p
    end
end
