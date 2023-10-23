@testset "Settings" begin
    spl = HMC(
        25,
        0.1;
        nchains = 10,
        integrator = "LF",
        sigma = [1.0],
    )

    sett = spl.settings
    hp = spl.hyperparameters
    dy = spl.hamiltonian_dynamics

    @test sett.nchains == 10
    @test sett.integrator == "LF"
    @test hp.ϵ == 0.1
    @test hp.N == 25
    @test hp.sigma == [1.0]
end

@testset "Partially_refresh_momentum" begin
    d = 10
    rng = MersenneTwister(0)
    u = MyHMC.Random_unit_vector(rng, d)
    @test length(u) == d
    @test isapprox(norm(u), 1.0, rtol = 0.0000001)
end

@testset "Init" begin
    d = 10
    m = zeros(d)
    s = Diagonal(ones(d))
    rng = MersenneTwister(1234)
    target = GaussianTarget(m, s)
    spl = HMC(0, 0.001)
    _, init = MyHMC.Step(rng, spl, target.h;
        trans_init_params = m)
    @test init.x == m
    @test init.g == m
    @test init.dE == 0.0
end

@testset "Step" begin
    d = 10
    m = zeros(d)
    s = Diagonal(ones(d))
    rng = MersenneTwister(1234)
    target = GaussianTarget(m, s)
    spl = HMC(25, 0.1; sigma = ones(d))
    _, init = MyHMC.Step(rng, spl, target.h;
        trans_init_params = target.prior_draw())
    _, step = MyHMC.Step(rng, spl, init)
    @test spl.hyperparameters.ϵ == 0.1
end
