config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using Random
using DelimitedFiles
using Distributions

function process_file(input_file, output_file)
    Random.seed!(0)
    raw = readdlm(input_file, ',')
    num_attributes = length(findall(x -> occursin("@ATTRIBUTE", string(x)), raw[:, 1])) - 2
    id_column = findfirst(x -> occursin("@ATTRIBUTE 'id'", string(x)), raw[:, 1]) - 1
    label_column = findfirst(x -> occursin("@ATTRIBUTE 'outlier'", string(x)), raw[:, 1]) - 1
    data_start_row = findlast(x -> x == "@DATA", raw[:, 1]) + 1
    raw[:, label_column] = map(x -> x == "'yes'" ? :outlier : :inlier, raw[:, label_column])
    data, labels = raw[data_start_row:end, [i for i in 1:size(raw, 2) if i != id_column && i != label_column]], raw[data_start_row:end, label_column]
    data = hcat(data, labels)
    @assert size(data, 2) - 1 == num_attributes
    @assert size(data, 1) == length(labels)
    if ADD_NOISE
        @info "Adding noise."
        data[:, 1:end-1] += rand(NOISE_DIST, size(data, 1), size(data, 2) - 1)
    end
    @info "Saving to '$output_file'."
    writedlm(output_file, data, ',')
    return nothing
end

ADD_NOISE = true
NOISE_DIST = Normal(0, 0.01)
target_versions_semantic = r"withoutdupl_norm_[0-9][0-9].arff"
target_versions_literature = r"withoutdupl_norm"
dataset_dir = normpath(data_raw_input_root)
output_path = normpath(data_input_root)

mkpath(output_path)
@info "Saving processed files to $output_path."

for dataset_class in ["semantic", "literature"]
    for d in raw_data_dirs[dataset_class]
        @info d
        outdir = joinpath(output_path, d)
        isdir(outdir) || mkpath(outdir)
        if dataset_class == "semantic"
            target_files = filter(x -> occursin(target_versions_semantic, x), readdir(joinpath(dataset_dir, dataset_class, d)))
            @assert length(target_files) == 1
            @info "[$(d)] Found $(length(target_files)) files."
            for f in target_files
               process_file(joinpath(dataset_dir, "semantic", d, f), joinpath(outdir, f[1:end-5] * ".csv"))
            end
        else
            target_files = filter(x -> occursin(target_versions_literature, x), readdir(joinpath(dataset_dir, dataset_class, d)))
            if (i = findfirst(x -> occursin("catremoved", x), target_files)) !== nothing
                target_file = target_files[i]
            else
                target_file = first(target_files)
            end
            @info "[$(d)] Processing '$target_file'."
            process_file(joinpath(dataset_dir, dataset_class, d, target_file), joinpath(outdir, target_file[1:end-5] * ".csv"))
        end
    end
end

@info "Preprocessing done."
