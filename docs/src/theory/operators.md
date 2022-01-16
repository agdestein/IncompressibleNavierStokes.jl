# Operators

```math
\mathbf{M} = 
```

Discrete gradient

```math
\mathbf{G} =
```

## Discretization of the Navier-Stokes equations

The discretized Navier-Stokes equations are given by

```math
\begin{align*}
M_h V_h(t) & = y_M \\
\Omega_h \frac{\mathrm{d} V_h}{\mathrm{d} t}(t) & = -C_h(V_h(t)) V_h(t) + \nu (D_h V_h(t) +
y_D) + f_h - (G_h p_h(t) + y_G)
\end{align*}
```
