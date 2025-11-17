# SpacecraftCA
Base tool bench for building spacecraft collision avoidance POMDPs, MDPs, and chance-constrained analysis.

## Purpose

This repository is a lightweight toolkit and example suite intended to help prototype spacecraft collision-avoidance models and (later) POMDP-style planners. The code contains basic state/action/transition/observation scaffolding, utility helpers for covariance propagation and collision probability estimation, and a few runnable examples.

### Notes for Vedant: Quick start (to run examples)


Heres a quick overview of what utilities currently exist in `src/utils/`

Prereqs:
- Julia (recommended 1.6+ or newer)

From your shell (zsh) in the project root:

```bash
cd /path/to/SpacecraftCA
# use the project environment and install dependencies (if any)
julia --project=.

# inside the Julia REPL you can run:
using Pkg
Pkg.instantiate()

# then run an example file
include("src/Examples/covariancePropExample.jl")
include("src/Examples/probabilityCollisionExample.jl")
```

Alternatively run the examples directly from the shell:

```bash
julia --project=. src/Examples/covariancePropExample.jl
julia --project=. src/Examples/probabilityCollisionExample.jl
```

## What is in this repo (high level)

- `src/` - main source files
	- `states.jl` - definitions and helpers for state representations
	- `actions.jl` - action set / manipulations (delta-v, maneuver descriptors)
	- `transition.jl` - deterministic/stochastic propagation / transition helpers
	- `observation.jl` - observation models or measurement helpers
	- `SpacecraftCA.jl` - package entry file (load / re-export useful symbols)
- `src/Examples/` - small, runnable examples demonstrating covariance propagation and collision-probability codepaths
- `src/utils/` - helpful utilities used by examples and sandbox code:
	- `propCovariance.jl` — covariance propagation utilities (linearized or simple dynamics propagation). Use to propagate state uncertainty forward in time.
	- `probabilityCollision.jl` — collision probability approximations and helpers (analytic approximations and wrappers around Monte Carlo estimates).
	- `genConjunctions.jl` — helpers to generate conjunction events / encounter geometries for batch processing.