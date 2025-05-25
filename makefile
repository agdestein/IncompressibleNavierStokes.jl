all: up

up:
	julia --project=. -e 'using Pkg; Pkg.update()'
	julia --project=docs -e 'using Pkg; Pkg.update()'
	julia --project=test -e 'using Pkg; Pkg.update()'
	julia --project=examples -e 'using Pkg; Pkg.update()'
	julia --project=scratch -e 'using Pkg; Pkg.update()'

.PHONY: docs
docs:
	julia --project=docs docs/make.jl

.PHONY: test
test:
	julia --project=test test/runtests.jl
