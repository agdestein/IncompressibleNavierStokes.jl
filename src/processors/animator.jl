"""
    animator(; setup, path, plot = fieldplot, nupdate = 1, kwargs...)

Animate a plot of the solution every `update` iteration.
The animation is saved to `path`, which should have one
of the following extensions:

- ".mkv"
- ".mp4"
- ".webm"
- ".gif"

The plot is determined by a `plotter` processor.
Additional `kwargs` are passed to Makie's `VideoStream`.
"""
animator(; setup, path, plot = fieldplot, nupdate = 1, kwargs...) =
    processor((stream, state) -> save(path, stream)) do outerstate
        ispath(dirname(path)) || mkpath(dirname(path))
        state = Observable(outerstate[])
        fig = plot(; setup, state)
        stream = VideoStream(fig; kwargs...)
        on(outerstate) do outerstate
            outerstate.n % nupdate == 0 || return
            state[] = outerstate
            recordframe!(stream)
        end
        stream
    end
