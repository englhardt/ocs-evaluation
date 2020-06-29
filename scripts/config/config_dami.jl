using JuMP, Gurobi, OneClassSampling, SVDD, OneClassActiveLearning
include(joinpath(@__DIR__, "config.jl"))

experiment_name="dami"

data_dirs = [x for x in data_dirs if x ∉ ["ALOI", "Annthyroid", "InternetAds", "KDDCup99", "PageBlocks", "PenDigits", "SpamBase", "Waveform", "Wilt"]]

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
quality_metrics = Dict(:mcc => matthews_corr, :kappa => cohens_kappa, :f1 => f1_score)

threshold_strategies = [
    OutlierPercentageThresholdStrategy()
]

model_init_strategy = Dict{Symbol, Any}(
                            :svdd => Dict{Symbol, Any}(
                                :ROT_CTax => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
                                :ROT_C1 => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.FixedCStrategy(1)),
                            ),
                            :sample_qe => Dict{Symbol, Any}(
                                :threshold_strategies => threshold_strategies
                        ))

KDE_GAMMA = :scott

competitors = [
    BPS(nothing, 0.05),
    DAEDS(30, 0.1, 0.3),
    DBSRSVDD(7, 0.5),
    HSR(20, 0.01),
    IESRSVDD(KDE_GAMMA, 0.5),
    FBPE(360),
    KFNCBD(100, 0.2),
    NDPSR(20, 10)
]

density_strategies = []
for t in threshold_strategies
    push!(density_strategies, RAPID(t, KDE_GAMMA))
end

sampling_strategies = Vector{Dict{Symbol, Any}}()

for c in competitors
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => c,
        :seed => 0
    ))
    for t in threshold_strategies
        push!(sampling_strategies, Dict{Symbol, Any}(
            :method => PreFilteringWrapper(c, KDE_GAMMA, t),
            :seed => 0
        ))
    end
end

for d in density_strategies
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => d,
        :seed => 0
    ))
end
