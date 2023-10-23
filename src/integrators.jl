function Leapfrog(sampler::MyHMCSampler, state::MyHMCState)
    ϵ = sampler.hyperparameters.ϵ
    sigma = sampler.hyperparameters.sigma
    h = state.h
    return Leapfrog(h, ϵ, sigma, state.x, state.u, state.l, state.g)
end

function Leapfrog(   
    h::Hamiltonian,
    ϵ::Number,
    sigma::AbstractVector,
    x::AbstractVector,
    u::AbstractVector,
    l::Number,
    g::AbstractVector,
)
    """leapfrog"""
    # go to the latent space
    z = x ./ sigma 
    
    #half step in momentum
    uu =  u .+ ((ϵ * 0.5).* (g .* sigma)) 
    #full step in x
    zz = z .+ (ϵ .* uu)
    xx = zz .* sigma # rotate back to parameter space
    ll, gg = -1 .* h.∂lπ∂θ(xx)
    #half step in momentum
    uu = uu .+ ((ϵ * 0.5).* (gg .* sigma)) 

    return xx, uu, ll, gg
end