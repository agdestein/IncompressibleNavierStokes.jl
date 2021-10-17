## Generation of a (non-)uniform mesh
function create_mesh!(setup)
    x, y = setup.grid.create_mesh(setup)

    ## Derived mesh quantities

    # Pressure positions
    xp = (x[1:end-1] + x[2:end]) / 2
    yp = (y[1:end-1] + y[2:end]) / 2

    # Distance between velocity points
    hx = diff(x)
    hy = diff(y)

    Nx = length(hx)
    Ny = length(hy)

    # Distance between pressure points
    gx = zeros(Nx + 1)
    gx[1] = hx[1] / 2
    gx[2:Nx] = (hx[1:Nx-1] + hx[2:Nx]) / 2
    gx[Nx+1] = hx[end] / 2

    gy = zeros(Ny + 1)
    gy[1] = hy[1] / 2
    gy[2:Ny] = (hy[1:Ny-1] + hy[2:Ny]) / 2
    gy[Ny+1] = hy[end] / 2

    @pack! setup.grid = x, y, xp, yp, hx, hy, gx, gy

    setup
end
