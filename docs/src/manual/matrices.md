```@meta
CurrentModule = IncompressibleNavierStokes
```

# Matrices

In IncompressibleNavierStokes, all operators are implemented as matrix-free kernels.
However, access to the underlying matrices can sometimes be useful, for example
to precompute matrix factorizations.
We therefore provide sparse matrix versions of some of the linear operators (see
full list [below](#API)).

## Example

Consider a simple setup

```@example Matrices
using IncompressibleNavierStokes
ax = range(0, 1, 17);
setup = Setup(; x = (ax, ax), Re = 1e3);
```

The matrices for the linear operators are named by appending `_mat` to the function name, for example:
`divergence`, `pressuregradient`, and `diffusion` become `divergence_mat`, `pressuregradient_mat`, `diffusion_mat` etc.

Let's assemble some matrices:

```@example Matrices
divergence_mat(setup)
```

```@example Matrices
pressuregradient_mat(setup)
```

```@example Matrices
diffusion_mat(setup)
```

Note the sparsity pattern with matrix
concatenation of two scalar operators for operators acting on or producing vector fields.
The `pressuregradient_mat` converts a scalar field to a vector field, and is thus the vertical concatenation of the matrices for ``\partial/\partial x`` and ``\partial/\partial y``,
while the `divergence_mat` is a horizontal concatenation of two similar matrices.
The periodic boundary conditions are not included in the operators above, and are implemented via their own matrix. The periodic extension is visible:


```@example Matrices
bc_u_mat(setup)
```

We can verify that the diffusion matrix gives the same results as the diffusion
kernel (without viscosity):

```@example Matrices
using Random
u = randn!(vectorfield(setup))
B = bc_u_mat(setup)
D = diffusion_mat(setup)
d_kernel = diffusion(apply_bc_u(u, 0.0, setup), setup; use_viscosity = false)
d_matrix = reshape(D * B * u[:], size(u))
maximum(abs, d_matrix - d_kernel)
```

Matrices only work on flattened fields `u[:]`, while the kernels work
on ``(D+1)``-array-shaped  fields for a dimension ``D \in \{2, 3\}``.

## Boundary conditions and matrices

Matrices can only be used to represent boundary conditions that depend linearly
on the input, such as periodic or Neumann boundary conditions.
Non-zero Dirichlet boundary conditions need to be accounted for separately.
Consider the following inflow-setup:

```@example Matrices
setup = Setup(;
    x = (ax, ax),
    boundary_conditions = (
        (DirichletBC((10.0, 0.0)), PressureBC()),
        (DirichletBC(), DirichletBC()),
    ),
)
```

We can assert that the kernel and matrix versions of the divergence give different results:

```@example Matrices
using Random
u = randn!(vectorfield(setup))
B = bc_u_mat(setup)
M = divergence_mat(setup)
div_kernel = divergence(apply_bc_u(u, 0.0, setup), setup)
div_matrix = reshape(M * B * u[:], size(div_kernel))
maximum(abs, div_matrix - div_kernel)
```

The solution is to create a vector containing the boundary conditions.
This is done by applying the BC kernel on an empty field:

```@example Matrices
uzero = zero(u)
yu = apply_bc_u(uzero, 0.0, setup)
yM = divergence(yu, setup)
```

By adding `yM`, we get the equality:

```@example Matrices
maximum(abs, (div_matrix + yM) - div_kernel)
```

## API

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["matrices.jl"]
```
