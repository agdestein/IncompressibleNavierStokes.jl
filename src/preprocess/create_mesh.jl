## generation of a (non-)uniform mesh
function create_mesh!(setup)
    x, y = setup.grid.create_mesh(setup);

    ## derived mesh quantities

    # pressure positions
    xp = (x[1:end-1]+x[2:end])/2;
    yp = (y[1:end-1]+y[2:end])/2;

    # distance between velocity points
    hx = diff(x);
    hy = diff(y);

    Nx = length(hx);
    Ny = length(hy);

    # distance between pressure points
    gx = zeros(Nx+1);
    gx[1] = hx[1]/2;
    gx[2:Nx] = (hx[1:Nx-1]+hx[2:Nx])/2;
    gx[Nx+1] = hx[end]/2;

    gy = zeros(Ny+1);
    gy[1] = hy[1]/2;
    gy[2:Ny] = (hy[1:Ny-1]+hy[2:Ny])/2;
    gy[Ny+1] = hy[end]/2;


    setup.grid.x = x;
    setup.grid.y = y;
    setup.grid.xp = xp;
    setup.grid.yp = yp;
    setup.grid.hx = hx;
    setup.grid.hy = hy;
    setup.grid.gx = gx;
    setup.grid.gy = gy;

    println("Nx = $Nx, minimum(hx) = $(minimum(hx))");
    println("Ny = $Ny, minimum(hy) = $(minimum(hy))");

    setup
end
