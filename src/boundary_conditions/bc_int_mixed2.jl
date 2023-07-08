function bc_int_mixed2(Nt, Nin, Nb, bc1, bc2, h1, h2)
    # total solution u is written as u = Bb*ub + Bin*uin
    # the boundary conditions can be written as Bbc*u = ybc
    # then u can be written entirely in terms of uin and ybc as:
    # u = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1)
    # Bb, Bin and Bbc depend on type of bc (Neumann/Dirichlet/periodic)

    # val1 and val2 can be scalars or vectors with either the value or the
    # derivative

    # (ghost) points on boundary / grid lines

    T = typeof(h1)

    # some input checking:
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    if Nb == 0 # no boundary points, so simply diagonal matrix without boundary contribution
        B1D = I(Nt)
        Btemp = spzeros(T, Nt, 2)
        ybc1 = zeros(T, 2)
        ybc2 = zeros(T, 2)
    else
        # boundary conditions
        Bbc = spzeros(T, Nb, Nt)
        ybc1_1D = zeros(T, Nb)
        ybc2_1D = zeros(T, Nb)

        if bc1 == :dirichlet && bc2 == :dirichlet
            # boundary matrices
            Bin = spdiagm(Nt, Nin, -2 => ones(T, Nin))
            Bb = spzeros(T, Nt, Nb)
            Bb[1:2, 1:2] = I(2)
            Bb[end-1:end, end-1:end] = I(2)

            Bbc[1, 1] = -1
            Bbc[1, 3] = 1
            Bbc[2, 2] = 1        # Dirichlet uLe
            ybc1_1D[1] = 0
            ybc1_1D[2] = 1

            Bbc[end, end] = -1
            Bbc[end, end-2] = 1
            Bbc[end-1, end-1] = 1
            ybc2_1D[end-1] = 1        # uRi
            ybc2_1D[end] = 0
        elseif bc1 == :periodic && bc2 == :periodic
            # boundary matrices
            Bin = spdiagm(Nt, Nin, -1 => ones(T, Nin))
            Bb = spzeros(T, Nt, Nb)
            Bb[1, 1] = 1
            Bb[end-1:end, end-1:end] = I(2)
            Bbc[1, 1] = -1
            Bbc[1, end-2] = 1
            Bbc[end-1, 2] = -1
            Bbc[end-1, end-1] = 1
            Bbc[end, 3] = -1
            Bbc[end, end] = 1
        else
            error("not implemented")
        end

        ybc1 = ybc1_1D
        ybc2 = ybc2_1D

        Btemp = Bb / (Bbc * Bb)
        B1D = Bin - Btemp * Bbc * Bin
    end

    (; B1D, Btemp, ybc1, ybc2)
end
