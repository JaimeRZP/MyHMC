mutable struct Hyperparameters
    ϵ::Float64
    N::Int
    sigma
end

Hyperparameters(;kwargs...) = begin
   ϵ = get(kwargs, :ϵ, 0.25)
   N = get(kwargs, :N, 25)
   sigma = get(kwargs, :sigma, [0.0])
   Hyperparameters(ϵ, N, sigma)
end

mutable struct Settings
    nchains::Int
    integrator::String
end

Settings(;kwargs...) = begin
    kwargs = Dict(kwargs)
    nchains = get(kwargs, :nchains, 1)
    integrator = get(kwargs, :integrator, "LF")
    Settings(nchains, integrator)
end

struct MyHMCSampler <: AbstractMCMC.AbstractSampler
    settings::Settings
    hyperparameters::Hyperparameters
    hamiltonian_dynamics::Function
end


"""
    MyHMC(
        nadapt::Int,
        TEV::Real;
        kwargs...
    )

Constructor for the MicroCanonical HMC sampler
"""
function HMC(N::Int, ϵ::Real; kwargs...)
    """the MyHMC (q = 0 Hamiltonian) sampler"""
    sett = Settings(; kwargs...)
    hyperparameters = Hyperparameters(; N=N, ϵ=ϵ, kwargs...)

    ### integrator ###
    if sett.integrator == "LF" # leapfrog
        hamiltonian_dynamics = Leapfrog
        grad_evals_per_step = 1.0
    elseif sett.integrator == "MN" # minimal norm
        hamiltonian_dynamics = Minimal_norm
        grad_evals_per_step = 2.0
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    return MyHMCSampler(sett, hyperparameters, hamiltonian_dynamics)
end

function Random_unit_vector(rng::AbstractRNG, d::Int; _normalize = true)
    """Generates a random (isotropic) unit vector."""
    u = randn(rng, d)
    if _normalize
        u = normalize(u)
    end
    return u
end

struct MyHMCState{T}
    rng::AbstractRNG
    i::Int
    x::Vector{T}
    u::Vector{T}
    l::T
    g::Vector{T}
    dE::T
    h::Hamiltonian
end

struct Transition{T}
    θ::Vector{T}
    δE::T
    ℓ::T
end

function Transition(state::MyHMCState, bijector)
    sample = bijector(state.x)[:]
    return Transition(sample, state.dE, -state.l)
end

function Step(
    rng::AbstractRNG,
    sampler::MyHMCSampler,
    h::Hamiltonian;
    bijector = NoTransform,
    trans_init_params = nothing,
    kwargs...,
)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = length(trans_init_params)
    l, g = -1 .* h.∂lπ∂θ(trans_init_params)
    u = Random_unit_vector(rng, d; _normalize=false)
    state = MyHMCState(rng, 0, trans_init_params, u, l, g, 0.0, h)
    transition = Transition(state, bijector)
    return transition, state
end

function Step(
    rng::AbstractRNG,
    sampler::MyHMCSampler,
    state::MyHMCState;
    bijector = NoTransform,
    kwargs...,
)
    local xx, uu, ll, gg, HH
    dialog = get(kwargs, :dialog, false)
    N = sampler.hyperparameters.N
    x, u, l, g, dE = state.x, state.u, state.l, state.g, state.dE
    H =  dot(u,u)/2 + l 
    # Hamiltonian step
    for i in 1:N
        xx, uu, ll, gg, HH = sampler.hamiltonian_dynamics(sampler, state)
    end
    #Metropolis Adjustment
    dEE =  HH - H
    accept = log(rand()) < dEE
    xx = @.(accept * xx + (1 - accept) * x)
    ll = @.(accept * ll + (1 - accept) * l)
    gg = @.(accept * gg + (1 - accept) * g)
    dEE = @.(accept * dEE + (1 - accept) * dE)
    # Resample energy
    uuu = Random_unit_vector(rng, length(uu); _normalize=false)

    state = MyHMCState(rng, state.i + 1, xx, uuu, ll, gg, dEE, state.h)
    transition = Transition(state, bijector)
    return transition, state
end

function Sample(
    sampler::MyHMCSampler,
    target::Target,
    n::Int;
    fol_name = ".",
    file_name = "samples",
    progress = true,
    kwargs...,
)
    return Sample(
        Random.GLOBAL_RNG,
        sampler,
        target,
        n;
        fol_name = fol_name,
        file_name = file_name,
        kwargs...,
    )
end

"""
    sample(
        rng::AbstractRNG,
        sampler::MyHMCSampler,
        target::Target,
        n::Int;
        init_params = nothing,
        fol_name = ".",
        file_name = "samples",
        progress = true,
        kwargs...
    )
Sampling routine
"""
function Sample(
    rng::AbstractRNG,
    sampler::MyHMCSampler,
    target::Target,
    n::Int;
    init_params = nothing,
    fol_name = ".",
    file_name = "samples",
    progress = true,
    kwargs...,
)
    io = open(joinpath(fol_name, "VarNames.txt"), "w") do io
        println(io, string(target.vsyms))
    end

    chain = []
    ### initial conditions ###
    if init_params != nothing
        @info "using provided init params"
        trans_init_params = target.inv_transform(init_params)
    else
        @info "sampling from prior"
        trans_init_params = target.prior_draw()
    end
    transition, state = Step(
        rng,
        sampler,
        target.h;
        bijector = target.transform,
        trans_init_params = trans_init_params,
        kwargs...,
    )
    push!(chain, [transition.θ; transition.δE; transition.ℓ])

    io = open(joinpath(fol_name, string(file_name, ".txt")), "w") do io
        println(io, [transition.θ; transition.δE; transition.ℓ])
        @showprogress "MyHMC: " (progress ? 1 : Inf) for i = 1:n-1
            try
                transition, state = Step(
                    rng,
                    sampler,
                    state;
                    bijector = target.transform,
                    kwargs...,
                )
                push!(chain, [transition.θ; transition.δE; transition.ℓ])
                println(io, [transition.θ; transition.δE; transition.ℓ])
            catch err
                if err isa InterruptException
                    rethrow(err)
                else
                    @warn "Divergence encountered after tuning"
                end
            end
        end
    end

    io = open(joinpath(fol_name, string(file_name, "_summary.txt")), "w") do io
        ess, rhat = Summarize(chain)
        println(io, ess)
        println(io, rhat)
    end

    return chain
end

function Summarize(samples::AbstractVector)
    _samples = zeros(length(samples), 1, length(samples[1]))
    _samples[:, 1, :] = mapreduce(permutedims, vcat, samples)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function Summarize(samples::AbstractMatrix)
    dim_a, dim_b = size(samples)
    _samples = zeros(dim_a, 1, dim_b)
    _samples[:, 1, :] = samples
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function Neff(samples, l::Int)
    ess, rhat = Summarize(samples)
    neff = ess ./ l
    return 1.0 / mean(1 ./ neff)
end