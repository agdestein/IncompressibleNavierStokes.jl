"""
    bc_general(Nt, Nin, Nb, bc1, bc2, h1, h2)

Total solution `u` is written as `u = Bb*ub + Bin*uin`

The boundary conditions can be written as `Bbc*u = ybc`

Then `u` can be written entirely in terms of `uin` and `ybc` as: `u =
(Bin-Btemp*Bbc*Bin)*uin + Btemp*ybc`, where `Btemp = Bb/(Bbc*Bb)`.

`Bb`, `Bin` and `Bbc` depend on type of bc (Neumann/Dirichlet/periodic) `val1` and `val2`
can be scalars or vectors with either the value or the derivative (ghost) points on
boundary/grid lines
"""
function bc_general(Nt, Nin, Nb, bc1, bc2, h1, h2)
    if Nt != Nin + Nb
        error("Number of inner points plus boundary points is not equal to total points")
    end

    T = typeof(h1)

    # Boundary conditions
    Bbc = spzeros(T, Nb, Nt)
    ybc1_1D = zeros(T, Nb)
    ybc2_1D = zeros(T, Nb)

    if Nb == 0
        # No boundary points, so simply diagonal matrix without boundary contribution
        B1D = I(Nt)
        Btemp = spzeros(T, Nt, 2)
        ybc1 = zeros(T, 2)
        ybc2 = zeros(T, 2)
    elseif Nb == 1
        # One boundary point
        Bb = spzeros(T, Nt, Nb)
        diagpos = -1
        if bc1 == :dirichlet
            Bbc[1, 1] = 1
            ybc1_1D[1] = 1        # uLe
            Bb[1, 1] = 1
        elseif bc1 == :pressure
            diagpos = 0
        elseif bc1 == :periodic
            diagpos = 0
            Bbc[1, 1] = -1
            Bbc[1, end] = 1
            Bb[end, 1] = 1
        else
            error("not implemented")
        end

        if bc2 == :dirichlet
            Bbc[Nb, end] = 1
            ybc2_1D[1] = 1        # uRi
            Bb[end, Nb] = 1
        elseif bc2 == :pressure

        elseif bc2 == :periodic # Actually redundant
            diagpos = 0
            Bbc[1, 1] = -1
            Bbc[1, end] = 1
            Bb[end, 1] = 1
        else
            error("not implemented")
        end

        # Boundary matrices
        Bin = spdiagm(Nt, Nin, diagpos => ones(T, Nin))
    elseif Nb == 2
        # Normal situation, 2 boundary points
        # Boundary matrices
        Bin = spdiagm(Nt, Nin, -1 => ones(T, Nin))
        Bb = spzeros(T, Nt, Nb)
        Bb[1, 1] = 1
        Bb[end, Nb] = 1

        if bc1 == :dirichlet
            Bbc[1, 1] = 1
            ybc1_1D[1] = 1        # uLe
        elseif bc1 == :pressure
            Bbc[1, 1] = -1
            Bbc[1, 2] = 1
            ybc1_1D[1] = 2 * h1     # DuLe
        elseif bc1 == :periodic
            Bbc[1, 1] = -1
            Bbc[1, end-1] = 1
            Bbc[Nb, 2] = -1
            Bbc[Nb, end] = 1
        else
            error("not implemented")
        end

        if bc2 == :dirichlet
            Bbc[Nb, end] = 1
            ybc2_1D[2] = 1        # uRi
        elseif bc2 == :pressure
            Bbc[Nb, end-1] = -1
            Bbc[Nb, end] = 1
            ybc2_1D[2] = 2 * h2     # duRi
        elseif bc2 == :periodic # Actually redundant
            Bbc[1, 1] = -1
            Bbc[1, end-1] = 1
            Bbc[Nb, 2] = -1
            Bbc[Nb, end] = 1
        else
            error("not implemented")
        end
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
