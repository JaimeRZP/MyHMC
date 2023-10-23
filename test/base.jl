@testset "Settings" begin
    spl = MyHMC(
        10_000,
        0.1;
        nchains = 10,
        integrator = "MN",
        eps = 0.1,
        L = 0.1,
        sigma = [1.0],
        gamma = 2.0,
        sigma_xi = 2.0,
        init_eps = 1.0,
        init_L = 1.0,
        init_sigma = [1.0],
    )

    sett = spl.settings
    hp = spl.hyperparameters
    dy = spl.hamiltonian_dynamics

    @test sett.nchains == 10
    @test sett.integrator == "MN"
    @test sett.TEV == 0.1
    @test sett.nadapt == 10_000
    @test sett.init_eps == 1.0
    @test sett.init_L == 1.0
    @test sett.init_sigma == [1.0]

    @test hp.eps == 0.1
    @test hp.L == 0.1
    @test hp.sigma == [1.0]
    @test hp.gamma == 2.0
    @test hp.sigma_xi == 2.0

    @test dy == MicroCanonicalHMC.Minimal_norm
end

@testset "Partially_refresh_momentum" begin
    d = 10
    rng = MersenneTwister(0)
    u = MicroCanonicalHMC.Random_unit_vector(rng, d)
    @test length(u) == d
    @test isapprox(norm(u), 1.0, rtol = 0.0000001)

    p = MicroCanonicalHMC.Partially_refresh_momentum(rng, 0.1, u)
    @test length(p) == d
    @test isapprox(norm(p), 1.0, rtol = 0.0000001)
end

@testset "Init" begin
    d = 10
    m = zeros(d)
    s = Diagonal(ones(d))
    rng = MersenneTwister(1234)
    target = GaussianTarget(m, s)
    spl = MyHMC(0, 0.001)
    _, init = MicroCanonicalHMC.Step(rng, spl, target.h;
        trans_init_params = m)
    @test init.x == m
    @test init.g == m
    @test init.dE == init.Feps == 0.0
    @test init.Weps == 1.0e-5
end

@testset "Step" begin
    d = 10
    m = zeros(d)
    s = Diagonal(ones(d))
    rng = MersenneTwister(1234)
    target = GaussianTarget(m, s)
    spl = MyHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d))
    aspl = MyHMC(0, 0.001; eps = 0.01, L = 0.1, sigma = ones(d), adaptive = true)
    _, init = MicroCanonicalHMC.Step(rng, spl, target.h;
        trans_init_params = target.prior_draw())
    tune_sigma, tune_eps, tune_L = MicroCanonicalHMC.tune_what(spl, d)
    tune_sigma, tune_eps, tune_L = MicroCanonicalHMC.tune_what(aspl, d)
    @test tune_sigma == tune_eps == tune_L == false
    _, step = MicroCanonicalHMC.Step(rng, spl, init)
    _, astep = MicroCanonicalHMC.Step(rng, aspl, init)
    @test spl.hyperparameters.eps == 0.01
    @test aspl.hyperparameters.eps != 0.01
    @test step.x == astep.x
end