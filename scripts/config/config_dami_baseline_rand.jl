using JuMP, Gurobi, OneClassSampling, SVDD, OneClassActiveLearning
include(joinpath(@__DIR__, "config.jl"))

experiment_name="dami_baseline_rand"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
quality_metrics = Dict(:mcc => matthews_corr, :kappa => cohens_kappa, :f1 => f1_score)

threshold_strategies = [
    OutlierPercentageThresholdStrategy()
]

model_init_strategy = Dict{Symbol, Any}(
                            :svdd => Dict{Symbol, Any}(
                                :ROT_CTax => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
                            ),
                            :sample_qe => Dict{Symbol, Any}(
                                :threshold_strategies => threshold_strategies
                        ))

NUM_RERUNS = 5
KDE_GAMMA = :scott

sampling_strategies = Vector{Dict{Symbol, Any}}()


for r in vcat(0.01:0.01:0.04, 0.05:0.05:1)
    for seed in 1:NUM_RERUNS
        push!(sampling_strategies, Dict{Symbol, Any}(
            :method => RandomRatioSampler(r),
            :seed => seed
        ))
    end
end
