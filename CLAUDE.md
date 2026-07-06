# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

IncompressibleNavierStokes.jl is a Julia package solving the incompressible
Navier-Stokes equations on a staggered Cartesian finite-volume grid, with
energy-conserving discretizations. Kernels are written with
KernelAbstractions.jl so the same code runs on CPU and GPU (CUDA). The package
is reverse-mode differentiable through ChainRulesCore rrules (Zygote) and
EnzymeCore rules, both backed by hand-written adjoint kernels.

## Commands

- **Tests** (full suite, ~3 min):
  `julia --project=test -e 'using Pkg; Pkg.resolve(); include("test/runtests.jl")'`
  The test project resolves the package via `[sources]` to the parent
  directory. Tests use TestItemRunner; `test/runtests.jl` filters test items
  to this directory (don't remove the filter — sibling checkouts are
  otherwise picked up). For quick iteration, use the persistent Julia session
  (julia-mcp) with `env_path` set to the repo root; Revise picks up edits.
- **Formatting**: `julia --project=@JuliaFormatter -e 'using JuliaFormatter; format(".")'`
  CI installs the *latest* JuliaFormatter release and fails on any diff, so
  keep the `@JuliaFormatter` shared environment fully updated
  (`Pkg.update()`, not just `Pkg.update("JuliaFormatter")` — old pinned deps
  hold the version back and older versions format differently).
- **Spell check**: `typos` (CLI, available locally). Config in `typos.toml`
  (single config file — extra `_typos.toml`/`.typos.toml` files shadow it).
- **Docs** are built with Documenter + DocumenterVitepress from `docs/`
  (`julia --project=docs docs/make.jl`). Set `INS_DOCS_FAST=1` for a fast
  build that skips executing all `@example` blocks.

## Docs gotchas

- Manual pages under `docs/src/manual/` use `@autodocs` blocks with `Pages`
  filters listing source files. A new `src/*.jl` file with documented symbols
  must be added to some page's `Pages` list, or the docs build fails.
- `@example` blocks in the manual (e.g. `matrices.md`, `sciml.md`,
  `getting_started.md`) are executed during the docs build — they are
  effectively integration tests. Keep them in sync with code changes, and
  run them locally (`julia --project=test`, which has Enzyme/Zygote) before
  pushing.
- `examples/*.jl` are Literate.jl scripts; the docs build converts them to
  pages and runs the ones marked `run = true`. The example list (titles,
  categories, gallery entries, sidebar) is defined *once* in `docs/make.jl`,
  which generates `docs/src/examples/index.md` and
  `docs/src/.vitepress/examples_sidebar.json` (both gitignored) —
  `docs/src/examples/generated/` is build output. New examples only need an
  entry in that list.

## Architecture notes

- `Setup(; x, boundary_conditions, backend, ...)` (src/grid.jl) returns a
  plain NamedTuple with grid quantities: `N` (volumes incl. ghosts), `Np`/`Ip`
  (pressure DOFs), `Nu`/`Iu` (velocity DOFs), `Δ`/`Δu` (volume widths),
  `xlims` (physical domain), `inside`, `backend`, `workgroupsize`.
- Staggered convention: `u[I, α]` lives on the right face of volume `I` in
  direction `α`; the divergence is a backward difference
  `(u[I, α] - u[I - e_α, α]) / Δ`. Fields include one layer of ghost volumes;
  `apply_bc_u!`/`apply_bc_p!` fill them.
- Operators (src/operators.jl) come in mutating (`divergence!`) and
  differentiable non-mutating (`divergence`) pairs. Adjoints are hand-written
  kernels (`*_adjoint!`) shared by the ChainRules rrules (end of
  operators.jl) and the Enzyme rules (ext/EnzymeCoreExt.jl). When testing
  AD rules, seed with *random* cotangents — all-ones or primal-output seeds
  can make wrong adjoints look right.
- Spectral quantities (src/spectral.jl): energy spectra, wavenumber shells,
  turbulence statistics. All RFFT-based; sums over wavenumbers must count the
  missing conjugate modes (`0 < k₁ < k₁ᵐᵃˣ` twice) — use the provided
  `spectralsum`/`energyinds` machinery. Requires a uniform periodic grid
  (`assert_uniform_periodic`). `random_field` constructs initial conditions
  that are exactly divergence-free on the staggered grid by projecting with
  the modified wavenumbers of the staggered divergence stencil.
- Time stepping (src/time_steppers/): explicit RK only. The user-facing
  entry point is `solve_unsteady(; setup, tlims, start, force!, params, ...)`
  with a `processors` NamedTuple for per-step output.
- Extensions (ext/): Makie plotting, Enzyme rules, AMGX/CUDSS pressure
  solvers — loaded conditionally on their trigger packages.

## Conventions

- The next release is breaking: renames and API changes are fine, no
  backwards-compatibility shims needed (status as of July 2026).
- Prefer several small, focused branches/PRs off `main` over one big branch.
- Match existing style: NamedTuples over structs for plain data, `α`/`β` for
  direction indices, `I`/`J` for CartesianIndex, kwargs-heavy APIs.
