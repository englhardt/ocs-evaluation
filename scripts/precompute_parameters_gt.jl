config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config_precompute_parameters_gt.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using SVDD, OneClassActiveLearning, Distributed, Gurobi, JuMP, MLKernels
using Serialization

include(joinpath(@__DIR__, "util", "setup_workers.jl"))

@info "Loading packages on all workers."
@everywhere using SVDD, OneClassActiveLearning, OneClassSampling, Gurobi, JuMP, MLBase, Memento, Random

@everywhere function Memento.debug(logger::Logger, error::ErrorException)
    Memento.warn(logger, "Caught ErrorException, msg='$(error.msg)')")
end

@everywhere function generate_fold_masks(labels, num_folds=10)
    folds = collect(StratifiedKfold(labels, num_folds))
    masks = []
    for k in 1:num_folds
        train_mask = [i ∈ folds[k] for i in 1:length(labels)]
        test_mask = .!train_mask
        push!(masks, (train_mask, test_mask))
    end
    return masks
end

@info "Searching for data sets in '$(data_input_root)'"
data_files = Vector{String}()
for data_set in vcat(values(data_dirs)...)
    data_dir = joinpath(data_input_root, data_set)
    if !isdir(data_dir)
        @info "Could not find dataset $data_dir"
        continue
    end
    local_names = readdir(data_dir)
    full_names = realpath.(joinpath.(data_input_root, data_dir, local_names))
    push!(data_files, full_names...)
end

@info "Initializing models."
all_model_params = @distributed vcat for file in data_files
    d, l = load_data(file)
    Random.seed!(0)
    fold_masks = generate_fold_masks(l, NUM_FOLDS)
    worker_params = []
    for k in 1:NUM_FOLDS
        @show file, k
        data, labels = d[:, fold_masks[k][1]], l[fold_masks[k][1]]
        n_obs = length(labels)
        p_out = sum(labels .== :outlier) / n_obs
        params = Dict{Symbol, Any}(
            :file => file,
            :n_observations => n_obs,
            :p_out => p_out,
            :fold => k,
            :train_mask => fold_masks[k][1],
            :test_mask => fold_masks[k][2]
        )
        # SVDD
        svdd_init = SVDD.SimpleCombinedStrategy(init_strategy[:gamma], SVDD.BoundedTaxErrorEstimate(p_out, 0.02, 0.98))
        model = instantiate(VanillaSVDD, data, fill(:U, n_obs), Dict{Symbol, Any}())
        initialize!(model, svdd_init)
        model_parameters = get_model_params(model)
        params[:svdd] = Dict{Symbol, Any}(
            :type => Symbol(typeof(model)),
            :C => get_model_params(model)[:C],
            :gamma => MLKernels.getvalue(model.kernel_fct.alpha)
        )
        # PWC
        pwc_gamma = init_strategy[:pwc][:gamma](data)
        pwc_threshold = calculate_threshold(init_strategy[:pwc][:threshold_strat], data, labels, pwc_gamma)
        params[:pwc] = Dict{Symbol, Any}(
            :gamma => pwc_gamma,
            :threshold => pwc_threshold,
            :threshold_strat => Symbol(typeof(init_strategy[:pwc][:threshold_strat]))
        )
        push!(worker_params, params)
    end
    worker_params
end

@info "Saving parameters."
model_params_by_filename = Dict{Tuple{String, Int}, Dict{Symbol, Any}}()
for model_params in vcat(all_model_params...)
    model_params_by_filename[basename(model_params[:file]), model_params[:fold]] = model_params
    delete!(model_params, :file)
end

serialize(data_parameter_file, model_params_by_filename)
@info "Done."
