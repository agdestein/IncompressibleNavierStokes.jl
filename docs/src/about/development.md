# Local development

## Use Juliaup

Install Julia using the [juliaup](https://julialang.org/downloads/) version manager.
This allows for choosing the Julia version, e.g. v1.11, by running

```
juliaup add 1.11
juliaup default 1.11
```

## Revise

It is recommended to use [Revise.jl](https://github.com/timholy/Revise.jl) for interactive development.
Add it to your global environment with

```
julia -e 'using Pkg; Pkg.add("Revise")'
```

and load it in the startup file (create the file and folder if it is not already there)
at `~/.julia/config/startup.jl` with

```
using Revise
```

Then changes to the IncompressibleNavierStokes modules are detected and reloaded live.

## Environments

To keep dependencies sparse, there are multiple `Project.toml` files in this
repository, specifying environments. For example, the
[`docs`](https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/main/docs/Project.toml)
environment contains packages that are required to build documentation, but not
needed to run the simulations.
To add local packages to an environment and be detectable by Revise, they need
to be `Pkg.develop`ed.  For example, the package
[NeuralClosure](https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/main/lib/NeuralClosure/Project.toml)
depends on IncompressibleNavierStokes, and IncompressibleNavierStokes needs to be `dev`ed with

```
julia --project=lib/NeuralClosure -e 'using Pkg; Pkg.develop(PackageSpec(; path = "."))'
```

Run this from the repository root, where `"."` is the path to IncompressibleNavierStokes.

On Julia v1.11, this linking is automatic, with the dedicated `[sources]`
sections in the `Project.toml` files. In that case, an environment can be
instantiated with

```
julia --project=lib/NeuralClosure -e 'using Pkg; Pkg.instantiate()'
```

etc., or interactively from the REPL with `] instantiate`.

### VSCode

In VSCode, you can choose an active environment by clicking on the `Julia env:` button in the status bar,
or press `ctrl`/`cmd` + `shift` + `p` and start typing `environment`:

- `Julia: Activate this environment` activates the one of the current open file
- `Julia: Change current environment` otherwise

Then scripts will be run from the selected environment.

### Environment vs package

If a `Project.toml` has a header with a `name` and `uuid`, then it is a package
with the module `src/ModuleName.jl`, and can be depended on in other projects
(by `add` or `dev`).
