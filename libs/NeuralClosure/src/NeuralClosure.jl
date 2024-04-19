"""
Neural closure modelling tools.
"""
module NeuralClosure

using CUDA
using ComponentArrays: ComponentArray
using Lux
using NNlib
using Tullio
using Zygote

include("closure.jl")
include("cnn.jl")
include("fno.jl")
include("training.jl")
include("create_les_data.jl")

export smagorinsky_closure
export cnn, fno, FourierLayer
export train
export mean_squared_error, create_relerr_prior, create_relerr_post
export create_loss_prior, create_loss_post
export create_dataloader_prior, create_dataloader_post
export create_callback, create_les_data, create_io_arrays
export wrappedclosure

end
