"""
    create_grid(T = Float64; Nx, Ny, xlims, ylims, stretch)

Create nonuniform cartesian box mesh `xlims` × `ylims` with stretch factors `stretch`.
"""
function create_grid(T = Float64; Nx, Ny, xlims, ylims, stretch)
    x = nonuniform_grid(xlims..., Nx, stretch[1])
    y = nonuniform_grid(ylims..., Ny, stretch[2])

    # Pressure positions
    xp = (x[1:end-1] + x[2:end]) / 2
    yp = (y[1:end-1] + y[2:end]) / 2

    # Distance between velocity points
    hx = diff(x)
    hy = diff(y)

    # Distance between pressure points
    gx = zeros(Nx + 1)
    gx[1] = hx[1] / 2
    gx[2:Nx] = (hx[1:Nx-1] + hx[2:Nx]) / 2
    gx[Nx+1] = hx[end] / 2

    gy = zeros(Ny + 1)
    gy[1] = hy[1] / 2
    gy[2:Ny] = (hy[1:Ny-1] + hy[2:Ny]) / 2
    gy[Ny+1] = hy[end] / 2

    Grid{T, 2}(; Nx, Ny, xlims, ylims, x, y, xp, yp, hx, hy, gx, gy)
end

"""
    create_grid(T = Float64; Nx, Ny, Nz, xlims, ylims, zlims, stretch)

Create nonuniform cartesian box mesh `xlims` × `ylims` × `zlims` with stretch factors
`stretch`.
"""
function create_grid(T = Float64; Nx, Ny, Nz, xlims, ylims, zlims, stretch)
    x = nonuniform_grid(xlims..., Nx, stretch[1])
    y = nonuniform_grid(ylims..., Ny, stretch[2])
    z = nonuniform_grid(zlims..., Nz, stretch[3])

    # Pressure positions
    xp = (x[1:end-1] + x[2:end]) / 2
    yp = (y[1:end-1] + y[2:end]) / 2
    zp = (z[1:end-1] + z[2:end]) / 2

    # Distance between velocity points
    hx = diff(x)
    hy = diff(y)
    hz = diff(z)

    # Distance between pressure points
    gx = zeros(Nx + 1)
    gx[1] = hx[1] / 2
    gx[2:Nx] = (hx[1:Nx-1] + hx[2:Nx]) / 2
    gx[Nx+1] = hx[end] / 2

    gy = zeros(Ny + 1)
    gy[1] = hy[1] / 2
    gy[2:Ny] = (hy[1:Ny-1] + hy[2:Ny]) / 2
    gy[Ny+1] = hy[end] / 2

    gz = zeros(Nz + 1)
    gz[1] = hz[1] / 2
    gz[2:Nz] = (hz[1:Nz-1] + hz[2:Nz]) / 2
    gz[Nz+1] = hz[end] / 2

    Grid{T, 3}(; Nx, Ny, Nz, xlims, ylims, zlims, x, y, z, xp, yp, zp, hx, hy, hz, gx, gy, gz)
end
