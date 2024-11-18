"""
Differential filter for the incompressible Navier-Stokes equations.

## Exports

The following symbols are exported by Filter:

$(EXPORTS)
"""

using LinearAlgebra
using SparseArrays

export differential_filter

"""
    differential_filter(u, v, w)

Applies a Helmholtz differential filter to the velocity fields (u, v, w).
Returns the filtered velocity fields (u_filtered, v_filtered, w_filtered).
"""
function differential_filter(u, setup, filter_radius, relax_parameter, lu_filter_mats)
    # Extract grid and boundary information from the setup
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Nu, Ip, Iu, Δ, Δu) = grid
    D = dimension()
	T = Float64

    # Apply filter across all velocity components (accounting for each dimension)
	ufilt = copy(u)

    for α = 1:D
		# old velocity (not filtered) only in internal points
		u_internal = view(u[Iu[α], α], :)
		# find new filtered velocity (when the diffusion matrix is applicable)
		ufilt_internal = lu_filter_mats[α] \ u_internal
		ufilt[Iu[α], α] .= reshape(ufilt_internal, size(ufilt[Iu[α], α]))
    end

	# Optional relaxation step
	if !isnothing(relax_parameter)
		u_filt .= relax(u, ufilt, relax_parameter)
    end

    return u_filt
end

function differential_filter!(u, setup, filter_radius, relax_parameter, lu_filter_mats)
    # Extract grid and boundary information from the setup
    (; grid, boundary_conditions) = setup
    (; dimension, x, N, Np, Nu, Ip, Iu, Δ, Δu) = grid
    D = dimension()
	T = Float64

	ufilt = copy(u)
    # Apply filter across all velocity components (accounting for each dimension)
    for α = 1:D
		# old velocity (not filtered) only in internal points
		u_internal = view(u[Iu[α], α], :)
		# find new filtered velocity (when the diffusion matrix is applicable)
		ufilt_internal = lu_filter_mats[α] \ u_internal
		ufilt[Iu[α], α] .= reshape(ufilt_internal, size(ufilt[Iu[α], α]))
    end

	# Optional relaxation step
	if !isnothing(relax_parameter)
		relax!(u, ufilt, relax_parameter)
    end
    return u
end

function decompose_filter_mat(setup, filter_radius)
	(; dimension, Nu) = setup.grid
	D = dimension()
    ntuple(D) do α
		L = diffusion_mat_velocity(setup, α)
		Id = sparse(I, prod(Nu[α]), prod(Nu[α]))
		filter_mat = Id - 2 * (filter_radius^2) * L
		lu(filter_mat)
	end
end

function relax(u, u_filtered, relax_parameter)
    u_relaxed = (1 - relax_parameter) .* u + relax_parameter .* u_filtered
    return u_relaxed
end

function relax!(u, u_filtered, relax_parameter)
    u .= (1 - relax_parameter) .* u + relax_parameter .* u_filtered
    return u
end
