isempty(ARGS) && error("Please pass a config file as command line argument.")
config_file = ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using SVDD, OneClassActiveLearning, OneClassSampling
using JuMP
using Gurobi
using MLKernels
using Memento
using Random
using Serialization

function setup_experiment_folder(config_file)
    isdir(exp_dir) || mkpath(exp_dir)
    mkpath(joinpath(exp_dir, "log", "experiment"))
    mkpath(joinpath(exp_dir, "log", "worker"))
    mkpath(joinpath(exp_dir, "results"))
    cp(config_file, joinpath(exp_dir, basename(config_file)))
    cp(joinpath(first(splitdir(config_file)), "config.jl"), joinpath(exp_dir, "config.jl"))
end

function create_experiments(sampling_strategies::Vector{Dict{Symbol, Any}}, solver::JuMP.OptimizerFactory, quality_metrics, model_init_strategy::Dict{Symbol, Any})::Vector{Dict{Symbol, Any}}
    experiment_configurations = [
        (data_file, model_params, s[:method], s[:seed])
        for s in sampling_strategies
        for data_file in data_files
        for model_params in values(filter(x -> x[1][1] == basename(data_file), model_params_by_filename))
    ]
    all_experiments = []
    for (data_file, model_params, sampling_strategy, seed) in experiment_configurations
        @show (data_file, model_params[:fold], sampling_strategy, seed)

        data_set_name = splitdir(splitdir(data_file)[1])[2]
        local_data_file_name = splitext(basename(data_file))[1]

        output_path = joinpath(exp_dir, "results", data_set_name)
        isdir(output_path) || mkpath(output_path)

        experiment = Dict{Symbol, Any}(
            :data_file => data_file,
            :data_set_name => data_set_name,
            :log_dir => joinpath(exp_dir, "log"),
            :models => Dict{Symbol, Any}(
                :svdd => Dict{Symbol, Any}(
                    :type => model_params[:svdd][:type],
                    :init_strategies => Dict{Symbol, Any}(
                        :svdd => SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(MLKernels.GaussianKernel(model_params[:svdd][:gamma])), SVDD.FixedCStrategy(model_params[:svdd][:C])),
                        [Symbol("svdd_reinit_$k") => v for (k, v) in model_init_strategy[:svdd]]...
                    ),
                    :solver => solver),
                :pwc => model_params[:pwc],
                :sample_qe => Dict{Symbol, Any}(
                    :gamma => model_params[:pwc][:gamma],
                    :threshold_strategies => model_init_strategy[:sample_qe][:threshold_strategies]
                )
            ),
            :train_mask => model_params[:train_mask],
            :test_mask => model_params[:test_mask],
            :fold => model_params[:fold],
            :sampling_strategy => deepcopy(sampling_strategy),
            :quality_metrics => quality_metrics,
            :seed => seed
        )
        exp_hash = hash(experiment)
        experiment[:hash] = "$(exp_hash)"
        experiment[:output_file] = joinpath(output_path, "$(local_data_file_name)_$(typeof(sampling_strategy))_VanillaSVDD_$(exp_hash).json")
        push!(all_experiments, experiment)
    end
    @info "Created $(realpath(exp_dir)) with $(length(all_experiments)) experiment settings."
    return all_experiments
end

function save_experiments(all_experiments::Array{Dict{Symbol, Any},1})
    exp_hashes_file = joinpath(exp_dir, "experiment_hashes")
    @info "Saving experiment hashes at $exp_hashes_file..."
    open(exp_hashes_file, "a") do f
        for e in all_experiments
            write(f, "$(e[:hash])\n")
        end
    end
    @info "... done."
    exp_filename = joinpath(exp_dir, "experiments.jser")
    serialize(exp_filename, all_experiments)
    @info "Experiments written to file $(exp_filename)."
end

# setup experiment directory
exp_dir = joinpath(data_output_root, experiment_name)
if isdir(exp_dir)
    print("Type 'yes' or 'y' to delete and overwrite experiment $(exp_dir): ")
    argin = readline()
    if argin == "yes" || argin == "y"
        rm(exp_dir, recursive=true)
    else
        error("Aborting...")
    end
end
mkpath(exp_dir)
exp_dir = realpath(exp_dir)

# find all data files
all(isdir.(joinpath.(data_input_root, data_dirs))) || error("Not all data dirs are valid.")
data_files = vcat(([joinpath.(data_input_root, x, readdir(joinpath(data_input_root, x))) for x in data_dirs])...)
@info "Found $(length(data_files)) data files."

# parse precompute parameters
isfile(data_parameter_file) || error("Precomputed parameters not found. Please follow the instructions in the Readme.")
@info "Using precompute parameters from file \"$data_parameter_file\"."
model_params_by_filename = deserialize(data_parameter_file)

# generate experiments
setup_experiment_folder(config_file)
Random.seed!(0)
experiments = create_experiments(sampling_strategies, solver, quality_metrics, model_init_strategy)
save_experiments(experiments)
