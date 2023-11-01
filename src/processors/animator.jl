"""
    animator(setup, path; nupdate = 1, plotter = field_plotter(setup); kwargs...)

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
animator(setup, path; nupdate = 1, plotter = field_plotter(setup), kwargs...) = processor(
    function (state)
        _state = Observable(state[])
        fig = plotter.initialize(_state)
        stream = VideoStream(fig; kwargs...)
        @lift begin
            _state[] = $state
            recordframe!(stream)
        end
        stream
    end;
    finalize = (stream, step_observer) -> save(path, stream),
    nupdate,
)
