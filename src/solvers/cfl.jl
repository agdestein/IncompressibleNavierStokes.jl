"""
    get_cfl_timestep!(buf, u, setup)

Get proposed maximum time step for convection and diffusion terms.
"""
function get_cfl_timestep!(buf, u, setup)
    (; Re, grid) = setup
    (; dimension, Δ, Δu, Iu) = grid
    D = dimension()

    # Initial maximum step size
    Δt = eltype(u[1])(Inf)

    # Check maximum step size in each dimension
    for α = 1:D
        # Diffusion
        Δαmin = minimum(view(Δu[α], Iu[α].indices[α]))
        Δt = min(Δt, Re * Δαmin^2 / 2)

        # Convection
        Δα = reshape(Δu[α], ntuple(Returns(1), α - 1)..., :)
        @. buf = Δα / abs(u[α])
        bmin = minimum(view(buf, Iu[α]))
        Δt = min(Δt, bmin)
    end

    Δt
end
