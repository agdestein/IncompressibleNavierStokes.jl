"""
    NeuralClosure

Neural closure modelling tools.

## Exports

$(EXPORTS)
"""
module NeuralClosure

using ComponentArrays: ComponentArray
using DocStringExtensions
using IncompressibleNavierStokes
using IncompressibleNavierStokes: Dimension, momentum!, apply_bc_u!, project!
using KernelAbstractions
using LinearAlgebra
using Lux
using Makie
using NNlib
using Observables
using Optimisers
using Random
using Zygote

# Put function signature in docstring by default
@template (FUNCTIONS, METHODS) = """
                                 $TYPEDSIGNATURES

                                 $DOCSTRING
                                 """

# Put type info in docstring by default
@template TYPES = """
                  $TYPEDEF

                  $FIELDS

                  $DOCSTRING
                  """

include("closure.jl")
include("cnn.jl")
include("fno.jl")
include("groupconv.jl")
include("training.jl")
include("filter.jl")
include("create_les_data.jl")

export cnn, gcnn, fno, FourierLayer, GroupConv2D, rot2, rot2stag
export train
export mean_squared_error,
    create_relerr_prior,
    create_relerr_post,
    create_relerr_symmetry_prior,
    create_relerr_symmetry_post
export create_loss_prior, create_loss_post
export create_dataloader_prior, create_dataloader_post
export create_callback, create_les_data, create_io_arrays
export wrappedclosure
export FaceAverage, VolumeAverage, reconstruct, reconstruct!

end
