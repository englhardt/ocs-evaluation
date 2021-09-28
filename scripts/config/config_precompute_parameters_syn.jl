include("config_syn.jl")

using JuMP, Gurobi, SVDD, OneClassSampling

NUM_FOLDS = 1
solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
init_strategy = Dict{Symbol, Any}(
    :svdd => SVDD.SimpleCombinedStrategy(SVDD.ModifiedMeanCriterion(), SVDD.BoundedTaxErrorEstimate(0.05, 0.02, 0.98)),
    :pwc => Dict(:gamma => SVDD.modified_mean_criterion, :threshold_strat => OutlierPercentageThresholdStrategy())
)
