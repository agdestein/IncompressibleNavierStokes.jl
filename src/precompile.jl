# Run tiny simulation to trigger precompilation
PrecompileTools.@compile_workload begin
    for D in (2, 3), T in (Float32, Float64)
        # Periodic
        x = ntuple(d -> range(T(0), T(1), 5), D)
        setup = Setup(; x, Re = T(1000))
        u = velocityfield(setup, (dim, x...) -> zero(x[1]))
        start = (; u)
        solve_unsteady(; start, setup, Δt = T(1e-3), tlims = (T(0), T(1e-2)))

        # Boundaries, temperature
        x = ntuple(d -> tanh_grid(T(0), T(1), 6), D)
        boundary_conditions = ntuple(d -> (DirichletBC(), PressureBC()), D)
        temperature = temperature_equation(;
            Pr = T(0.71),
            Ra = T(1e6),
            Ge = T(1.0),
            boundary_conditions,
        )
        setup = Setup(; x, Re = T(1000), temperature, boundary_conditions)
        u = velocityfield(setup, (dim, x...) -> zero(x[1]))
        temp = temperaturefield(setup, (x...) -> zero(x[1]))
        start = (; u, temp)
        solve_unsteady(; start, setup, Δt = T(1e-3), tlims = (T(0), T(1e-2)))
    end
end
