using JuMP, Gurobi, OneClassSampling, SVDD, OneClassActiveLearning
include(joinpath(@__DIR__, "config.jl"))

experiment_name="dami_baseline_gt"
data_parameter_file = joinpath(data_input_root, "parameters_gt.jser")

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
quality_metrics = Dict(:mcc => matthews_corr, :kappa => cohens_kappa, :f1 => f1_score,
                       :auc => [0.01, 0.02, 0.05, 0.1, 0.2])

model_init_strategy = Dict{Symbol, Any}(
                            :svdd => Dict{Symbol, Any}(),
                            :sample_qe => Dict{Symbol, Any}(
                                :threshold_strategies => []
                        ))


sampling_strategies = Vector{Dict{Symbol, Any}}()

push!(sampling_strategies, Dict{Symbol, Any}(
    :method => RandomRatioSampler(1.0),
    :seed => 0
))
