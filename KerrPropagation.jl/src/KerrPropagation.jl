module KerrPropagation

using Reexport
@reexport using CUDA
using FFTW,CUDA.CUFFT,LinearAlgebra

include("dft_utils.jl")

include("solver.jl")
export kerrPropagation

end 
