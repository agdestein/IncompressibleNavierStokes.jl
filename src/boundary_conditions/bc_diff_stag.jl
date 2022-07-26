function bc_diff_stag(Nt, Nin, Nb, h1, h2)
    # Total solution u is written as u = Bb*ub + Bin*uin
    # The boundary conditions can be written as Bbc*u = ybc
    # Then u can be written entirely in terms of uin and ybc as:
    # u = (Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc, where
    # Btemp = Bb*(Bbc*Bb)^(-1)
    # Bb, Bin and Bbc depend on type of bc (Neumann/Dirichlet/periodic)


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
        B1D = I(Nt)
        Btemp = spzeros(Nt, 2)
        ybc1 = zeros(2)
        ybc2 = zeros(2)
    elseif Nb == 1
        # One boundary point (should not be unnecessary)
    elseif Nb == 2
        # Normal situation, 2 boundary points

        # Boundary matrices
        Bin = spdiagm(Nt, Nin, -1 => ones(Nin))
        Bb = spzeros(Nt, Nb)
        Bb[1, 1] = 1
        Bb[end, Nb] = 1

        Bbc[1, 1] = -1
        Bbc[1, end-1] = 1
        Bbc[2, 2] = -1
        Bbc[2, end] = 1

        Bbc[1, 1] = -1
        Bbc[1, end-1] = 1
        Bbc[2, 2] = -1
        Bbc[2, end] = 1
    end

    if Nb ∈ (1, 2)
        ybc1 = ybc1_1D
        ybc2 = ybc2_1D

        Btemp = Bb / (Bbc * Bb)
        B1D = Bin - Btemp * Bbc * Bin
    end

    (; B1D, Btemp, ybc1, ybc2)
end
