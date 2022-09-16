# Convert example scripts to Jupyter notebooks

using Literate

# Scripts to convert
examples = [
    "Actuator2D.jl"
    "Actuator3D.jl"
    "BackwardFacingStep2D.jl"
    "BackwardFacingStep3D.jl"
    "DecayingTurbulence2D.jl"
    "DecayingTurbulence3D.jl"
    "LidDrivenCavity2D.jl"
    "LidDrivenCavity3D.jl"
    "PlanarMixing2D.jl"
    "ShearLayer2D.jl"
    "TaylorGreenVortex2D.jl"
    "TaylorGreenVortex3D.jl"
]

# Location of notebooks
output_dir = "notebooks"
ispath(output_dir) || mkpath(output_dir)

# Convert scripts (set `execute = true` to run notebooks upon conversion)
for e ∈ examples
    Literate.notebook(e, output_dir; execute = false)
end
