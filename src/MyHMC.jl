module MyHMC

export Settings, HMC, Sample
export Summarize
export GaussianTarget, RosenbrockTarget, CustomTarget
export ParallelTarget

using LinearAlgebra, Statistics, Random, DataFrames
using DynamicPPL, LogDensityProblemsAD, LogDensityProblems, ForwardDiff
using AbstractMCMC, MCMCChains, MCMCDiagnosticTools, Distributed
using Distributions, DistributionsAD, ProgressMeter

abstract type Target <: AbstractMCMC.AbstractModel end

include("hamiltonian.jl")
include("targets.jl")
include("sampler.jl")
include("integrators.jl")
include("abstractmcmc.jl")

end
