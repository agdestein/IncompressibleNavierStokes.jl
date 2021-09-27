function BC_general(Nt, Nin, Nb, BC1, BC2, h1, h2)
    # total solution u is written as u = Bb*ub + Bin*uin
    # the boundary conditions can be written as Bbc*u = ybc
    # then u can be written entirely in terms of uin and ybc as:
    # u = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1)
    # Bb, Bin and Bbc depend on type of BC (Neumann/Dirichlet/periodic)
    # val1 and val2 can be scalars or vectors with either the value or the
    # derivative
    # (ghost) points on boundary / grid lines

    # some input checking:
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    if Nb == 0
        # no boundary points, so simply diagonal matrix without boundary contribution
        B1D = speye(Nt)
        Btemp = sparse(Nt, 2)
        ybc1 = zeros(2, 1)
        ybc2 = zeros(2, 1)
        BC.B1D = B1D
        BC.Btemp = Btemp
        BC.ybc1 = ybc1
        BC.ybc2 = ybc2
        return BC
    end

    # boundary conditions
    Bbc = sparse(Nb, Nt)
    ybc1_1D = zeros(Nb)
    ybc2_1D = zeros(Nb)

    if Nb == 2
        # normal situation, 2 boundary points

        # boundary matrices
        Bin = spdiagm(Nt, Nin, -1 => ones(Nt))
        Bb = sparse(Nt, Nb)
        Bb[1, 1] = 1
        Bb[end, Nb] = 1

        if BC1 == "dir"
            Bbc[1, 1] = 1
            ybc1_1D[1] = 1        # uLe
        elseif BC1 == "pres"
            Bbc[1, 1] = -1
            Bbc[1, 2] = 1
            ybc1_1D[1] = 2 * h1     # duLe
        elseif BC1 == "per"
            Bbc[1, 1] = -1
            Bbc[1, end-1] = 1
            Bbc[Nb, 2] = -1
            Bbc[Nb, end] = 1
        else
            error("not implemented")
        end

        if BC2 == "dir"
            Bbc[Nb, end] = 1
            ybc2_1D[2] = 1        # uRi
        elseif BC2 == "pres"
            Bbc[Nb, end-1] = -1
            Bbc[Nb, end] = 1
            ybc2_1D[2] = 2 * h2     # duRi
        elseif BC2 == "per" # actually redundant
            Bbc[1, 1] = -1
            Bbc[1, end-1] = 1
            Bbc[Nb, 2] = -1
            Bbc[Nb, end] = 1
        else
            error("not implemented")
        end

    elseif Nb == 1
        # one boundary point

        Bb = sparse(Nt, Nb)

        diagpos = -1
        if BC1 == "dir"
            Bbc[1, 1] = 1
            ybc1_1D[1] = 1        # uLe
            Bb[1, 1] = 1
        elseif BC1 == "pres"
            diagpos = 0
        elseif BC1 == "per"
            diagpos = 0
            Bbc[1, 1] = -1
            Bbc[1, end] = 1
            Bb[end, 1] = 1
        else
            error("not implemented")
        end

        if BC2 == "dir"
            Bbc[Nb, end] = 1
            ybc2_1D[1] = 1        # uRi
            Bb[end, Nb] = 1
        elseif BC2 == "pres"

        elseif BC2 == "per" # actually redundant
            diagpos = 0
            Bbc[1, 1] = -1
            Bbc[1, end] = 1
            Bb[end, 1] = 1
        else
            error("not implemented")
        end

        # boundary matrices
        Bin = spdiagm(Nt, Nin, diagpos => ones(Nt))

    end

    ybc1 = ybc1_1D
    ybc2 = ybc2_1D

    Btemp = Bb * (Bbc * Bb \ sparse(I, Nb, Nb)) # = inv(Bbc*Bb)
    B1D = Bin - Btemp * Bbc * Bin

    BC.B1D = B1D
    BC.Btemp = Btemp
    BC.ybc1 = ybc1
    BC.ybc2 = ybc2

    BC
end
