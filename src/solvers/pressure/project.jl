function project(solver, u, setup)
    (; Ω) = setup.grid
    T = eltype(u[1])

    # Divergence of tentative velocity field
    M = divergence(u, setup)
    M = @. M * Ω

    # Solve the Poisson equation
    p = poisson(solver, M)
    p = apply_bc_p(p, T(0), setup)

    # Compute pressure correction term
    G = pressuregradient(p, setup)

    # Update velocity, which is now divergence free
    u .- G
end

function project!(solver, u, setup; M, p, G)
    (; Ω) = setup.grid
    T = eltype(u[1])

    # Divergence of tentative velocity field
    divergence!(M, u, setup)
    @. M *= Ω

    # Solve the Poisson equation
    poisson!(solver, p, M)
    apply_bc_p!(p, T(0), setup)

    # Compute pressure correction term
    pressuregradient!(G, p, setup)

    # Update velocity, which is now divergence free
    ntuple(length(u)) do α
        @. u[α] -= G[α]
    end
end
