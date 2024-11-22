```@meta
CurrentModule = IncompressibleNavierStokes
```

# Using IncompressibleNavierStokes in SciML

The [SciML organization](https://sciml.ai/) is a collection of tools for solving equations and modeling systems. It has a coherent development principle, unified APIs over large collections of equation solvers, pervasive differentiability and sensitivity analysis, and features many of the highest performance and parallel implementations one can find.

In particular, [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) contains tools to solve differential equations defined as $\dfrac{du}{dt} = f(u, t)$ that include a large collection of solvers, sensitivity analysis, and more.

Using IncompressibleNavierStokes it is possible to write the momentum equations without the pressure by explicitly solving the discrete Poisson equation and obtaining:

```math
\begin{align*}
\frac{\mathrm{d} u_h}{\mathrm{d} t} &= (I - G L^{-1} W M)
(F(u_h) - y_G) - G L^{-1} W \frac{\mathrm{d} y_M}{\mathrm{d} t}\\ &=f(u_h).
\end{align*}
```

The derivation and the drawbacks of this approach are discussed in the [documentation](/docs/src/manual/spatial.md).

This projected right-hand side can be used in the SciML solvers to solve the Navier-Stokes equations. The following example shows how to use the SciML solvers to solve the ODEs obtained from the Navier-Stokes equations.

```julia
using DifferentialEquations 
f(u, p, t) = create_right_hand_side(setup, psolver) 
u0 = INITIAL_CONDITION
tspan = (0.0, 1.0)     # time span where to solve.
problem = ODEProblem(f, u0, tspan) #SciMLBase.ODEProblem
sol = solve(problem, Tsit5(), reltol = 1e-8, abstol = 1e-8) # sol: SciMLBase.ODESolution
```

Alternatively, it is also possible to use an [in-place formulation](https://docs.sciml.ai/DiffEqDocs/stable/basics/problem/#In-place-vs-Out-of-Place-Function-Definition-Forms) 

```julia
f(du,u,p,t) = right_hand_side!(du, u, Ref([setup, psolver]), t)
```
that is usually faster than the out-of-place formulation.

You can look [here](https://docs.sciml.ai/DiffEqDocs/stable/basics/overview/) for more information on how to use the SciML solvers and all the options available.

## API 
```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["sciml.jl"]
```