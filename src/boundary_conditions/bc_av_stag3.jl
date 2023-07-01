function bc_av_stag3(Nt, Nin, Nb, bc1, bc2, h1, h2)
    # total solution u is written as u = Bb*ub + Bin*uin
    # the boundary conditions can be written as Bbc*u = ybc
    # then u can be written entirely in terms of uin and ybc as:
    # u = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1)
    # Bb, Bin and Bbc depend on type of bc (Neumann/Dirichlet/periodic)

    # val1 and val2 can be scalars or vectors with either the value or the
    # derivative

    # (ghost) points on staggered locations (pressure lines)

    T = typeof(h1)

    # some input checking:
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    if Nb == 0
        # no boundary points, so simply diagonal matrix without boundary contribution
        B1D = I(Nt)
        Btemp = spzeros(T, Nt, 2)
        ybc1 = zeros(T, 2)
        ybc2 = zeros(T, 2)
    else
        # boundary conditions
        Bbc = spzeros(T, Nb, Nt)
        ybc1_1D = zeros(T, Nb)
        ybc2_1D = zeros(T, Nb)

        if Nb == 6
            # normal situation, 2 boundary points

            # boundary matrices
            Bin = spdiagm(Nt, Nin, -3 => ones(T, Nin))
            Bb = spzeros(T, Nt, Nb)
            Bb[1:3, 1:3] = I(3)
            Bb[(end-2):end, (end-2):end] = I(3)

            if bc1 == :dirichlet
                Bbc[1, 1] = -1
                Bbc[1, 6] = 1
                Bbc[2, 2] = 1 / 2      # Dirichlet uLo
                Bbc[2, 5] = 1 / 2
                Bbc[3, 3] = -1
                Bbc[3, 4] = 1
                ybc1_1D[1] = 0
                ybc1_1D[2] = 1
                ybc1_1D[3] = 0
            elseif bc1 == :symmetric
                error("not implemented")
            elseif bc1 == :periodic
                Bbc[1:3, 1:3] = -I(3)
                Bbc[1:3, (end-5):(end-3)] = I(3)
                Bbc[(end-2):end, 4:6] = -I(3)
                Bbc[(end-2):end, (end-2):end] = I(3)
            else
                error("not implemented")
            end

            if bc2 == :dirichlet
                Bbc[end, end] = -1
                Bbc[end, end-5] = 1
                Bbc[end-1, end-1] = 1 / 2
                Bbc[end-1, end-4] = 1 / 2
                Bbc[end-2, end-2] = -1
                Bbc[end-2, end-3] = 1
                ybc2_1D[end-2] = 0
                ybc2_1D[end-1] = 1
                ybc2_1D[end] = 0
            elseif bc2 == :symmetric
                error("not implemented")
            elseif bc2 == :periodic
                # actually redundant
                Bbc[1:3, 1:3] = -I(3)
                Bbc[1:3, (end-5):(end-3)] = I(3)
                Bbc[(end-2):end, 4:6] = -I(3)
                Bbc[(end-2):end, (end-2):end] = I(3)
            else
                error("not implemented")
            end
        elseif Nb == 1  # one boundary point
            error("not implemented")
        end

        ybc1 = ybc1_1D
        ybc2 = ybc2_1D

        Btemp = Bb / (Bbc * Bb)
        B1D = Bin - Btemp * Bbc * Bin
    end

    (; B1D, Btemp, ybc1, ybc2)
end
