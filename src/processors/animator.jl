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
Additional `kwargs` are passed to `plot`.
"""
animator(; setup, path, plot = fieldplot, nupdate = 1, framerate = 24, visible = true, kwargs...) =
    processor((stream, state) -> save(path, stream)) do outerstate
        ispath(dirname(path)) || mkpath(dirname(path))
        state = Observable(outerstate[])
        fig = plot(state; setup, kwargs...)
        visible && display(fig)
        stream = VideoStream(fig; framerate, visible)
        on(outerstate) do outerstate
            outerstate.n % nupdate == 0 || return
            state[] = outerstate
            recordframe!(stream)
        end
        stream
    end
