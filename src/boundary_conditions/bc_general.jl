"""
    bc_general(Nt, Nin, Nb, h1, h2)

Total solution `u` is written as `u = Bb*ub + Bin*uin`

The boundary conditions can be written as `Bbc*u = ybc`

Then `u` can be written entirely in terms of `uin` and `ybc` as: `u =
(Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc`, where `Btemp = Bb/(Bbc*Bb)`.

`Bb`, `Bin` and `Bbc` depend on type of bc (Neumann/Dirichlet/periodic) `val1` and `val2`
can be scalars or vectors with either the value or the derivative (ghost) points on
boundary/grid lines
"""
function bc_general(Nt, Nin, Nb, h1, h2)
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
        # One boundary point
        Bb = spzeros(Nt, Nb)
        diagpos = -1
        diagpos = 0
        Bbc[1, 1] = -1
        Bbc[1, end] = 1
        Bb[end, 1] = 1

        diagpos = 0
        Bbc[1, 1] = -1
        Bbc[1, end] = 1
        Bb[end, 1] = 1

        # Boundary matrices
        Bin = spdiagm(Nt, Nin, diagpos => ones(Nin))
    elseif Nb == 2
        # Normal situation, 2 boundary points
        # Boundary matrices
        Bin = spdiagm(Nt, Nin, -1 => ones(Nin))
        Bb = spzeros(Nt, Nb)
        Bb[1, 1] = 1
        Bb[end, Nb] = 1

        Bbc[1, 1] = -1
        Bbc[1, end-1] = 1
        Bbc[Nb, 2] = -1
        Bbc[Nb, end] = 1

        Bbc[1, 1] = -1
        Bbc[1, end-1] = 1
        Bbc[Nb, 2] = -1
        Bbc[Nb, end] = 1
    else
        error("Nb must be 0, 1, or 2")
    end

    if Nb ∈ (1, 2)
        ybc1 = ybc1_1D
        ybc2 = ybc2_1D

        Btemp = Bb / (Bbc * Bb)
        B1D = Bin - Btemp * Bbc * Bin
    end

    (; B1D, Btemp, ybc1, ybc2)
end
