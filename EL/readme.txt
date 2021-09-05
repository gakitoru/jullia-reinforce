[setup] don't use conda
https://github.com/JuliaML/OpenAIGym.jl

[terminal]
conda deactivate
conda deactivate
alias julia=/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia
julia
ENV["PYTHON"] = "/usr/bin/python3"
include("target.jl")