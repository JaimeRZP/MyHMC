NoTransform(x) = x

mutable struct CustomTarget <: Target
    d::Int
    vsyms::Any
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CustomTarget(nlogp, grad_nlogp, priors; kwargs...) = begin
    d = length(priors)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    function prior_draw()
        x = [rand(dist) for dist in priors]
        xt = transform(x)
        return xt
    end
    hamiltonian = Hamiltonian(nlogp, grad_nlogp)
    CustomTarget(d, hamiltonian, NoTransform, NoTransform, prior_draw)
end

mutable struct GaussianTarget <: Target
    d::Int
    vsyms::Any
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

GaussianTarget(_mean::AbstractVector, _cov::AbstractMatrix) = begin
    d = length(_mean)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    _gaussian = MvNormal(_mean, _cov)
    ℓπ(θ::AbstractVector) = logpdf(_gaussian, θ)
    ∂lπ∂θ(θ::AbstractVector) = (logpdf(_gaussian, θ), gradlogpdf(_gaussian, θ))
    hamiltonian = Hamiltonian(ℓπ, ∂lπ∂θ)

    function prior_draw()
        xt = rand(MvNormal(zeros(d), ones(d)))
        return xt
    end

    GaussianTarget(d, vsyms, hamiltonian, NoTransform, NoTransform, prior_draw)
end

mutable struct RosenbrockTarget <: Target
    d::Int
    vsyms::Any
    h::Hamiltonian
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

RosenbrockTarget(a, b; kwargs...) = begin
    kwargs = Dict(kwargs)
    d = kwargs[:d]
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    function ℓπ(x; a = a, b = b)
        x1 = x[1:Int(d / 2)]
        x2 = x[Int(d / 2)+1:end]
        m = @.((a - x1)^2 + b * (x2 - x1^2)^2)
        return 0.5 * sum(m)
    end

    function ∂lπ∂θ(x)
        return ℓπ(x), ForwardDiff.gradient(ℓπ, x)
    end

    hamiltonian = Hamiltonian(ℓπ, ∂lπ∂θ)

    function prior_draw()
        x = rand(MvNormal(zeros(d), ones(d)))
        return x
    end

    RosenbrockTarget(d, vsyms, hamiltonian, NoTransform, NoTransform, prior_draw)
end
