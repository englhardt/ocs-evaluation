using JuMP, Gurobi, OneClassSampling, SVDD, OneClassActiveLearning
include(joinpath(@__DIR__, "config_syn.jl"))

experiment_name="synthetic"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
quality_metrics = Dict(:mcc => matthews_corr, :kappa => cohens_kappa, :f1 => f1_score)

threshold_strategies = [
    GroundTruthThresholdStrategy(),
]

model_init_strategy = Dict{Symbol, Any}(
                            :svdd => Dict{Symbol, Any}(
                                :MMC_C1 => SVDD.SimpleCombinedStrategy(SVDD.ModifiedMeanCriterion(), SVDD.FixedCStrategy(1)),
                            ),
                            :sample_qe => Dict{Symbol, Any}(
                                :threshold_strategies => threshold_strategies
                        ))

KDE_GAMMA = :mod_mean_crit

competitors = [
    BPS(nothing, 0.005),
    DAEDS(30, 0.1, 0.3),
    DBSRSVDD(7, 0.3),
    HSR(20, 0.01),
    IESRSVDD(KDE_GAMMA, 0.5),
    FBPE(360),
    KFNCBD(100, 0.2),
    NDPSR(20, 7)
]

density_sampler = [
    RAPID(OutlierPercentageThresholdStrategy(0.0), KDE_GAMMA),
]

sampling_strategies = Vector{Dict{Symbol, Any}}()

for c in vcat(competitors, density_sampler)
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => PreFilteringWrapper(c, KDE_GAMMA, GroundTruthThresholdStrategy()),
        :seed => 0
    ))
end
