config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config_syn.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using Distributions

using DelimitedFiles
using Random

function make_blobs(n_inliers=100, n_dims=2; n_outliers=0, n_clusters=3, cluster_std=0.05, threshold=1, data_range=(0, 1))
    center_box = (data_range[1] + (data_range[2] - data_range[1]) * 0.2,
                  data_range[1] + (data_range[2] - data_range[1]) * 0.8)
    centers = rand(Uniform(center_box...), (n_dims, n_clusters))
    dists = [MvNormal(centers[:, i], cluster_std) for i in 1:n_clusters]
    n_samples_per_center = fill(div(n_inliers, n_clusters), n_clusters)
    for i in 1:n_inliers % n_clusters
        n_samples_per_center[i] += 1
    end
    data_inliers = []
    for (i, d) in enumerate(dists)
        data_d = zeros(n_dims, 0)
        while size(data_d, 2) < n_samples_per_center[i]
            candidates = rand(d, (n_samples_per_center[i]) * 2)
            mask = pdf(d, candidates) .>= threshold
            candidates = candidates[:, mask]
            data_d = hcat(data_d, candidates[:, 1:min(end, n_samples_per_center[i] - size(data_d, 2))])
        end
        @assert size(data_d) == (n_dims, n_samples_per_center[i])
        push!(data_inliers, data_d)
    end
    data_inliers = hcat(data_inliers...)

    mix_dist = MixtureModel(dists)
    data_outliers = zeros(n_dims, 0)
    while size(data_outliers, 2) < n_outliers
        candidates = rand(Uniform(data_range...), (n_dims, n_outliers * 2))
        mask = pdf(mix_dist, candidates) .< threshold
        candidates = candidates[:, mask]
        data_outliers = hcat(data_outliers, candidates[:, 1:min(end, n_outliers - size(data_outliers, 2))])
    end
    d = hcat(data_inliers, data_outliers)
    d .-= minimum(d, dims=2)
    d ./= maximum(d, dims=2)
    data_inliers, data_outliers = d[:, 1:end-n_outliers], d[:, end-n_outliers+1:end]
    @assert size(data_inliers) == (n_dims, n_inliers)
    @assert size(data_outliers) == (n_dims, n_outliers)
    return data_inliers, data_outliers
end

function generate_data(n_in, dim, n_out, n_clusters, seed, name="standard")
    Random.seed!(seed)
    d_in, d_out = make_blobs(n_in, dim, n_outliers=n_out, n_clusters=n_clusters)
    d = hcat(d_in, d_out)
    l = vcat(fill(:inlier, n_in), fill(:outlier, n_out))
    folder = joinpath(data_input_root, name)
    mkpath(folder)
    file_name = "syn_$(name)_$(n_in)_in_$(n_out)_out_$(dim)_dim_$(n_clusters)_clusters_$(seed)_seed.csv"
    println(file_name)
    open(joinpath(folder, file_name), "w") do f
       writedlm(f, [d' l], ',')
    end
    return nothing
end

NUM_GENS = 5

# vary dims
N_CLUSTERS = 5
N_IN = 1000
N_OUT = 1000
DIMS = [10, 50, 100, 200, 500]
for SEED in 1:NUM_GENS
    for DIM in DIMS
        generate_data(N_IN, DIM, N_OUT, N_CLUSTERS, SEED, "dims")
    end
end

# vary size
SEED = 0
DIM = 50
N_CLUSTERS = 5
N_OUT = 1000
SIZES = [100, 500, 1000, 2500, 5000, 7500, 10000]
for SEED in 1:NUM_GENS
    for N_IN in SIZES
        generate_data(N_IN, DIM, N_OUT, N_CLUSTERS, SEED, "sizes")
    end
end

# vary clusters
DIM = 50
N_IN = 1000
N_OUT = 1000
CLUSTERS = [1, 3, 5, 10, 20]
for SEED in 1:NUM_GENS
    for N_CLUSTERS in CLUSTERS
        generate_data(N_IN, DIM, N_OUT, N_CLUSTERS, SEED, "clusters")
    end
end
