"""
Neural closure modelling tools.
"""
module NeuralClosure

using ComponentArrays: ComponentArray
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

include("closure.jl")
include("cnn.jl")
include("fno.jl")
include("groupconv.jl")
include("training.jl")
include("filter.jl")
include("create_les_data.jl")

export cnn, gcnn, fno, FourierLayer, GroupConv2D, rot2
export train
export mean_squared_error, create_relerr_prior, create_relerr_post
export create_loss_prior, create_loss_post
export create_dataloader_prior, create_dataloader_post
export create_callback, create_les_data, create_io_arrays
export wrappedclosure
export FaceAverage, VolumeAverage, reconstruct, reconstruct!

end
