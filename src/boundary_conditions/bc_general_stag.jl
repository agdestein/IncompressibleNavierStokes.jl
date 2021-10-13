function bc_general_stag(Nt, Nin, Nb, bc1, bc2, h1, h2)
    # Total solution u is written as u = Bb*ub + Bin*uin
    # The boundary conditions can be written as Bbc*u = ybc
    # Then u can be written entirely in terms of uin and ybc as:
    # U = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1]
    # Bb, Bin and Bbc depend on type of BC (Neumann/Dirichlet/periodic)
    # Val1 and val2 can be scalars or vectors with either the value or the
    # Derivative
    # (ghost) points on staggered locations (pressure lines)

    # Some input checking:
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    # Boundary conditions
    Bbc = spzeros(Nb, Nt)
    ybc1_1D = zeros(Nb)
    ybc2_1D = zeros(Nb)

    if Nb == 0
        # No boundary points, so simply diagonal matrix without boundary contribution
        B1D = sparse(I, Nt, Nt)
        Btemp = spzeros(Nt, 2)
        ybc1 = zeros(2, 1)
        ybc2 = zeros(2, 1)
    elseif Nb == 1
        # One boundary point
        Bb = spzeros(Nt, Nb)

        diagpos = -1
        if bc1 == :dirichlet
            Bbc[1, 1] = 1 / 2
            Bbc[1, 2] = 1 / 2
            ybc1_1D[1] = 1 # ULe
            Bb[1, 1] = 1
        elseif bc1 == :symmetric
            Bbc[1, 1] = -1
            Bbc[1, 2] = 1
            ybc1_1D[1] = h1 # DuLo
        elseif bc1 == :periodic
            Bbc[1, 1] = -1
            Bbc[1, end] = 1
            Bb[1, 1] = 1
        else
            error("not implemented")
        end

        if bc2 == :dirichlet
            Bbc[Nb, end-1] = 1 / 2
            Bbc[Nb, end] = 1 / 2
            ybc2_1D[1] = 1 # URi
            Bb[end, Nb] = 1
        elseif bc2 == :symmetric
            Bbc[Nb, end-1] = -1
            Bbc[Nb, end] = 1
            ybc2_1D[1] = h2 # DuUp
        elseif bc2 == :periodic
            Bbc[1, 1] = -1
            Bbc[1, end] = 1
            Bb[1, 1] = 1
        else
            error("not implemented")
        end

        # Boundary matrices
        Bin = spdiagm(Nt, Nin, diagpos => ones(Nin))
    elseif Nb == 2
        # Normal situation, 2 boundary points
        # Boundary matrices
        Bin = spdiagm(Nt, Nin, -1 => ones(Nin))
        Bb = spzeros(Nt, Nb)
        Bb[1, 1] = 1
        Bb[end, Nb] = 1

        if bc1 == :dirichlet
            Bbc[1, 1] = 1 / 2
            Bbc[1, 2] = 1 / 2
            ybc1_1D[1] = 1 # ULo
        elseif bc1 == :symmetric
            Bbc[1, 1] = -1
            Bbc[1, 2] = 1
            ybc1_1D[1] = h1 # DuLo
        elseif bc1 == :periodic
            Bbc[1, 1] = -1
            Bbc[1, end-1] = 1
            Bbc[2, 2] = -1
            Bbc[2, end] = 1
        else
            error("not implemented")
        end

        if bc2 == :dirichlet
            Bbc[end, end-1] = 1 / 2
            Bbc[end, end] = 1 / 2
            ybc2_1D[2] = 1 # UUp
        elseif bc2 == :symmetric
            Bbc[2, end-1] = -1
            Bbc[2, end] = 1
            ybc2_1D[2] = h2 # DuUp
        elseif bc2 == :periodic
            Bbc[1, 1] = -1
            Bbc[1, end-1] = 1
            Bbc[2, 2] = -1
            Bbc[2, end] = 1
        else
            error("not implemented")
        end
    else
        error("Nb must be 0, 1, or 2")
    end

    if Nb ∈ [1, 2]
        ybc1 = ybc1_1D
        ybc2 = ybc2_1D
        Btemp = Bb * (Bbc * Bb \ sparse(I, Nb, Nb)) # = inv(Bbc*Bb)
        B1D = Bin - Btemp * Bbc * Bin
    end

    (; B1D, Btemp, ybc1, ybc2)
end
