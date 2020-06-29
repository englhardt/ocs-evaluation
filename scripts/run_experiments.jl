config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using SVDD, OneClassActiveLearning
using OneClassSampling
using Memento, Gurobi, JSON, JuMP
using Random
using Distributed
using MLKernels
using Serialization
using Dates

@info "Running experiment in \"$(realpath(data_output_root))\"."
include(joinpath(@__DIR__, "util", "setup_workers.jl"))

@info "Loading packages on all workers."
@everywhere using Pkg
@everywhere using SVDD, OneClassActiveLearning, OneClassSampling, JSON, JuMP, Gurobi, Memento, Random, MLKernels, Serialization
@everywhere import SVDD: SVDDneg
@everywhere include(joinpath(@__DIR__, "util", "evaluate.jl"))
@info "Loaded."

@everywhere fmt_string = "[{name} | {date} | {level}]: {msg}"
@everywhere loglevel = "debug"

@everywhere function setup_logging(experiment)
    setlevel!(getlogger("root"), "error")
    setlevel!(getlogger(OneClassActiveLearning), loglevel)
    setlevel!(getlogger(SVDD), loglevel)

    exp_logfile = joinpath(experiment[:log_dir], "experiment", "$(experiment[:hash]).log")
    worker_logfile = joinpath(experiment[:log_dir], "worker", "$(gethostname())_$(getpid()).log")

    WORKER_LOGGER = Memento.config!("runner", "debug"; fmt=fmt_string)

    exp_handler = DefaultHandler(exp_logfile, DefaultFormatter(fmt_string))
    push!(getlogger(OneClassActiveLearning), exp_handler, experiment[:hash])
    push!(getlogger(SVDD), exp_handler, experiment[:hash])
    push!(WORKER_LOGGER, exp_handler, experiment[:hash])

    worker_handler = DefaultHandler(worker_logfile, DefaultFormatter(fmt_string))
    setlevel!(gethandlers(WORKER_LOGGER)["console"], "error")
    push!(WORKER_LOGGER, worker_handler)

    return WORKER_LOGGER
end

@everywhere function cleanup_logging(worker_logger::Logger, experiment_hash)
    delete!(gethandlers(getlogger(OneClassActiveLearning)), experiment_hash)
    delete!(gethandlers(getlogger(SVDD)), experiment_hash)
    delete!(gethandlers(worker_logger), experiment_hash)
    return nothing
end

@everywhere function Memento.warn(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::ErrorException)
    Memento.warn(logger, "Caught ErrorException, msg='$(error.msg)')")
end

@everywhere function run_experiment(experiment::Dict, warmup=false)
    Random.seed!(experiment[:seed])

    WORKER_LOGGER = setup_logging(experiment)
    info(WORKER_LOGGER, "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash])")
    if isfile(experiment[:output_file])
        warn(WORKER_LOGGER, "Aborting experiment because the output file already exists. Filename: $(experiment[:output_file])")
        cleanup_logging(WORKER_LOGGER, experiment[:hash])
        return nothing
    end
    e = deepcopy(experiment)

    data, labels = load_data(e[:data_file])
    e[:data_stats] = Dict(:n_observations => size(data, 2),
                            :n_dims => size(data, 1),
                            :outlier_ratio => sum(labels .== :outlier) / size(data, 2))

    errorfile = joinpath(e[:log_dir], "worker", "$(gethostname())_$(getpid())")
    try
        time_sampling = @elapsed sample_mask = OneClassSampling.sample(e[:sampling_strategy], data, labels)
        e[:result] = Dict{Symbol, Any}(:sample_size => sum(sample_mask), :time_sampling => time_sampling)
        if e[:result][:sample_size] > SAMPLE_SIZE_LIMIT
            throw(TooLargeSampleException(e[:result][:sample_size]))
        end
        @show e[:data_file]
        @show e[:sampling_strategy]
        @show e[:result][:sample_size]
        eval_dict = evaluate(e[:sampling_strategy], e[:models],
                             sample_mask,
                             data, labels, e[:quality_metrics])
        e[:result] = merge(e[:result], eval_dict)
        e[:exit_code] = :success
    catch ex
        e[:exit_code] = Symbol(typeof(ex))
        if isa(e, SamplingException)
            warn(WORKER_LOGGER, "Experiment $(e[:hash]) failed during sampling.")
            warn(WORKER_LOGGER, ex)
        elseif !isa(e, EmptySampleException)
            @warn "Experiment $(e[:hash]) finished with unkown error."
            @warn ex
            @warn stacktrace(catch_backtrace())
            warn(WORKER_LOGGER, "Experiment $(e[:hash]) finished with unkown error.")
            warn(WORKER_LOGGER, ex)
        end
        warn(WORKER_LOGGER, "Experiment $(e[:hash]) finished with unkown error.")
        warn(WORKER_LOGGER, ex)
    finally
        # Clean up for result file writing
        delete!(e, :quality_metrics)
        e[:sampling_strategy] = string(e[:sampling_strategy])
        e[:models][:sample_qe][:threshold_strategies] = string(e[:models][:sample_qe][:threshold_strategies])
        if e[:exit_code] != :success
            info(WORKER_LOGGER, "Writing error hash to $errorfile.error.")
            open("$errorfile.error", "a") do f
                print(f, "$(e[:hash])\n")
            end
        end
        if !WARMUP
            info(WORKER_LOGGER, "Writing result to $(e[:output_file]).")
            open(e[:output_file], "w") do f
                JSON.print(f, e)
            end
        end
        cleanup_logging(WORKER_LOGGER, e[:hash])
    end

    return nothing
end

all_experiments = []
for s in readdir(data_output_root)
    exp_dir = joinpath(data_output_root, s)
    if !isdir(exp_dir) || occursin("warmup", s)
        @info "skipping $s"
        continue
    end
    @info "Running experiments in directory $s"
    @info "Loading experiments.jser"
    experiments = deserialize(open(joinpath(exp_dir, "experiments.jser")))
    append!(all_experiments, experiments)
end

shuffle!(all_experiments)

@info "Warming up workers."
try
    @everywhere begin
        WARMUP = true
        include($config_file)
        warmup_experiments = deserialize(open(joinpath(data_output_root, "warmup", "experiments.jser")))
        print("Starting warmup.")
        map(run_experiment, warmup_experiments)
        print("Finished warmup.")
    end
catch e
    @warn e
    rmprocs(workers())
    @info "Warmup failed. Terminating."
    exit()
end
@info "Warmup done."

@everywhere WARMUP = false

try
    @info "Running $(length(all_experiments)) experiments."
    start = now()
    @info "$(start) - Running experiments..."
    pmap(run_experiment, all_experiments, on_error=ex->print("!!! ", ex))
    finish = now()
    @info "$finish - Done."
    @info "Ran $(length(all_experiments)) experiment(s) in $(canonicalize(Dates.CompoundPeriod(finish - start)))"
catch e
    @warn e
finally
    rmprocs(workers())
end
