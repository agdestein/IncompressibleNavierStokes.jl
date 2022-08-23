# Operators

All discrete operators are represented as sparse matrices.

## Discretization of the Navier-Stokes equations

The discretized Navier-Stokes equations are given by

```math
\begin{align*}
\mathbf{M} \mathbf{V}(t) & = \mathbf{y}_{\mathbf{M}} \\ \mathbf{\Omega}
\frac{\mathrm{d} \mathbf{V}}{\mathrm{d} t}(t) & = -\mathbf{C}(\mathbf{V}(t))
\mathbf{V}(t) + \nu (\mathbf{D} \mathbf{V}(t) + \mathbf{y}_{\mathbf{D}}) +
\mathbf{f} - (\mathbf{G} \mathbf{p}(t) + \mathbf{y}_{\mathbf{G}})
\end{align*}
```
