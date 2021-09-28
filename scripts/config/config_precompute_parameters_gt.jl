include("config.jl")

using JuMP, Gurobi, SVDD, OneClassSampling

NUM_FOLDS = 5
solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
data_parameter_file = joinpath(data_input_root, "parameters_gt.jser")

init_strategy = Dict{Symbol, Any}(
    :gamma => SVDD.RuleOfThumbScott(),
    :svdd => SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
    :pwc => Dict(:gamma => SVDD.rule_of_scott, :threshold_strat => OutlierPercentageThresholdStrategy())
)
