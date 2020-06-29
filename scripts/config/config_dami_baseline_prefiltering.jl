using JuMP, Gurobi, OneClassSampling, SVDD, OneClassActiveLearning
include(joinpath(@__DIR__, "config.jl"))

experiment_name="dami_baseline_prefiltering"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
quality_metrics = Dict(:mcc => matthews_corr, :kappa => cohens_kappa, :f1 => f1_score)

threshold_strategies = [
    OutlierPercentageThresholdStrategy()
]

model_init_strategy = Dict{Symbol, Any}(
                            :svdd => Dict{Symbol, Any}(
                                :ROT_C1 => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.FixedCStrategy(1)),
                            ),
                            :sample_qe => Dict{Symbol, Any}(
                                :threshold_strategies => threshold_strategies
                        ))

KDE_GAMMA = :scott

sampling_strategies = Vector{Dict{Symbol, Any}}()

for t in threshold_strategies
    push!(sampling_strategies, Dict{Symbol, Any}(
        :method => HDS(1.0, KDE_GAMMA, t),
        :seed => 0
    ))
end
