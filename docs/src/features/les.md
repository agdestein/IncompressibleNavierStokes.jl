# Large eddy simulation

Depending on the problem specification, a given grid resolution may not be
sufficient to resolve all spatial features of the flow. If refining the grid is
too costly, a closure model can be used to predict the sub-grid stresses. The
following eddy viscosity models are available:

- [`SmagorinskyModel`](@ref)
- [`QRModel`](@ref)
- [`MixingLengthModel`](@ref)

These models add a local contribution to the global baseline viscosity. This
non-constant field is computed from the local velocity field.

In addition, the default [`LaminarModel`](@ref) assumes that there are no
sub-grid stresses, and has the advantage of having a constant diffusion
operator.
