function bc_int_mixed_stag3(Nt, Nin, Nb, bc1, bc2, h1, h2)
    # total solution u is written as u = Bb*ub + Bin*uin
    # the boundary conditions can be written as Bbc*u = ybc
    # then u can be written entirely in terms of uin and ybc as:
    # u = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1)
    # Bb, Bin and Bbc depend on type of bc (Neumann/Dirichlet/periodic)

    # val1 and val2 can be scalars or vectors with either the value or the
    # derivative

    # (ghost) points on staggered locations (pressure lines)

    # some input checking:
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    if Nb == 0 # no boundary points, so simply diagonal matrix without boundary contribution
        B1D = I(Nt)
        Btemp = spzeros(Nt, 2)
        ybc1 = zeros(2)
        ybc2 = zeros(2)
    else
        # boundary conditions
        Bbc = spzeros(Nb, Nt)
        ybc1_1D = zeros(Nb)
        ybc2_1D = zeros(Nb)

        if Nb == 4
            # normal situation, 2 boundary points

            # boundary matrices
            Bin = spdiagm(Nt, Nin, -2 => ones(Nin))
            Bb = spzeros(Nt, Nb)
            Bb[1:2, 1:2] = I(2)
            Bb[end-1:end, end-1:end] = I(2)

            if bc1 == :dirichlet
                Bbc[1, 1] = -1  # Neumann type for skew-symmetry
                Bbc[1, 4] = 1
                Bbc[2, 2] = -1
                Bbc[2, 3] = 1
                ybc1_1D[1] = 0
                ybc1_1D[2] = 0
            elseif bc1 == :symmetric
                error("not implemented")
            elseif bc1 == :periodic
                Bbc[1:2, 1:2] = -I(2)
                Bbc[1:2, end-3:end-2] = I(2)
                Bbc[end-1:end, 3:4] = -I(2)
                Bbc[end-1:end, end-1:end] = I(2)
            else
                error("not implemented")
            end

            if bc2 == :dirichlet
                Bbc[end, end] = -1  # Neumann type for skew-symmetry
                Bbc[end, end-3] = 1
                Bbc[end-1, end-1] = -1
                Bbc[end-1, end-2] = 1
                ybc2_1D[end-1] = 0
                ybc2_1D[end] = 0
            elseif bc2 == :symmetric
                error("not implemented")
            elseif bc2 == :periodic # actually redundant
                Bbc[1:2, 1:2] = -I(2)
                Bbc[1:2, end-3:end-2] = I(2)
                Bbc[end-1:end, 3:4] = -I(2)
                Bbc[end-1:end, end-1:end] = I(2)
            else
                error("not implemented")
            end
        elseif Nb == 1
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
