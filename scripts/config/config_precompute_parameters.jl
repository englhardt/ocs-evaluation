include("config.jl")

using JuMP, Gurobi, SVDD, OneClassSampling

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
init_strategy = Dict{Symbol, Any}(
    :svdd => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
    :pwc => Dict(:gamma => SVDD.rule_of_scott, :threshold_strat => OutlierPercentageThresholdStrategy())
)
