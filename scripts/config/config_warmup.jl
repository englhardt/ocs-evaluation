using JuMP, Gurobi, OneClassSampling, SVDD, OneClassActiveLearning
include(joinpath(@__DIR__, "config.jl"))

experiment_name="warmup"

data_dirs = ["Parkinson"]

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
quality_metrics = Dict(:mcc => matthews_corr, :kappa => cohens_kappa, :f1 => f1_score)

threshold_strategies = [
    GroundTruthThresholdStrategy(),
    OutlierPercentageThresholdStrategy(),
    OutlierPercentageThresholdStrategy(0.05),
]

model_init_strategy = Dict{Symbol, Any}(
                            :svdd => Dict{Symbol, Any}(
                                :ROT_CTax => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
                                :ROT_C1 => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.FixedCStrategy(1)),
                            ),
                            :sample_qe => Dict{Symbol, Any}(
                                :threshold_strategies => threshold_strategies
                        ))
NUM_RERUNS = 1
KDE_GAMMA = :scott

baselines = [
    RandomRatioSampler(0.1),
    HDS(0.1, KDE_GAMMA, OutlierPercentageThresholdStrategy())
]

competitors = [
    BPS(nothing, 0.05),
    DAEDS(30, 0.1, 0.3),
    DBSRSVDD(7, 0.3),
    HSR(5, 0.01),
    IESRSVDD(KDE_GAMMA, 0.5),
    FBPE(360),
    KFNCBD(100, 0.2),
    NDPSR(20, 10)
]

density_strategies = [
    RAPID(OutlierPercentageThresholdStrategy(), KDE_GAMMA),
]

sampling_strategies = Vector{Dict{Symbol, Any}}()

for b in baselines
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => b,
        :seed => 0
    ))
end

for c in competitors
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => c,
        :seed => 0
    ))
end

for t in threshold_strategies
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => PreFilteringWrapper(HSR(5, 0.1), KDE_GAMMA, t),
        :seed => 0
    ))
end

for d in density_strategies
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => d,
        :seed => 0
    ))
end
