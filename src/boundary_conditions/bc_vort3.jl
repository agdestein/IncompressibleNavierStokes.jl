function bc_vort3(Nt, Nin, Nb, bc1, bc2, h1, h2)
    # total solution u is written as u = Bb*ub + Bin*uin
    # the boundary conditions can be written as Bbc*u = ybc
    # then u can be written entirely in terms of uin and ybc as:
    # u = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1)
    # Bb, Bin and Bbc depend on type of bc (Neumann/Dirichlet/periodic)

    # val1 and val2 can be scalars or vectors with either the value or the
    # derivative

    # (ghost) points on boundary / grid lines

    # to calculate vorticity; two ghost points at lower/left boundary, one
    # ghost point upper/right boundary

    # some input checking:
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    if Nb == 0
        # no boundary points, so simply diagonal matrix without boundary contribution
        B1D = I(Nt)
        Btemp = spzeros(Nt, 2)
        ybc1 = zeros(2)
        ybc2 = zeros(2)
    else
        # boundary conditions
        Bbc = spzeros(Nb, Nt)
        ybc1_1D = zeros(Nb)
        ybc2_1D = zeros(Nb)

        if Nb == 3
            # normal situation, 4 boundary points

            # boundary matrices
            Bin = spdiagm(Nt, Nin, -2 => ones(Nin))
            Bb = spzeros(Nt, Nb)
            Bb[1:2, 1:2] = I(2)
            Bb[end, end] = 1
            if bc1 == :dirichlet
                error("not implemented")
            elseif bc1 == :pressure
                error("not implemented")
            elseif bc1 == :periodic
                Bbc[1:2, 1:2] = -I(2)
                Bbc[1:2, end-2:end-1] = I(2)
                Bbc[end, 3] = -1
                Bbc[end, end] = 1
            else
                error("not implemented")
            end

            if bc2 == :dirichlet
                error("not implemented")
            elseif bc2 == :pressure
                error("not implemented")
            elseif bc2 == :periodic
                # actually redundant
                Bbc[1:2, 1:2] = -I(2)
                Bbc[1:2, end-2:end-1] = I(2)
                Bbc[end, 3] = -1
                Bbc[end, end] = 1
            else
                error("not implemented")
            end
        else
            # one boundary point
            error("not implemented")
        end

        ybc1 = ybc1_1D
        ybc2 = ybc2_1D

        Btemp = Bb / (Bbc * Bb)
        B1D = Bin - Btemp * Bbc * Bin
    end

    (; B1D, Btemp, ybc1, ybc2)
end
