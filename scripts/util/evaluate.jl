function train_svdd_model(model::DataType, init_strategy, solver,
                          data::Array{Float64, 2}, labels::Vector{Symbol})
    model = model(data)
    SVDD.initialize!(model, init_strategy)
    set_adjust_K!(model, true)
    SVDD.fit!(model, solver)
    return model
end

function predict_svdd_model(model, test_data)
    return SVDD.classify.(SVDD.predict(model, test_data))
end

function predict_score_svdd_model(model, test_data)
    pred_scores = SVDD.predict(model, test_data)
    return SVDD.classify.(pred_scores), pred_scores
end

function evaluate_with_svdd(model::DataType, init_strategy, solver,
                            data::Array{Float64, 2}, labels::Vector{Symbol},
                            test_data::Array{Float64, 2}, test_labels::Vector{Symbol},
                            quality_metrics)
    try
        time_train = @elapsed model = train_svdd_model(model, init_strategy, solver, data, labels)
        time_pred = @elapsed pred, pred_scores = predict_score_svdd_model(model, test_data)
        scores = evaluate_prediction(quality_metrics, test_labels, pred, pred_scores)
        gamma = MLKernels.getvalue(model.kernel_fct.alpha)
        C = model.C
        num_support_vectors = length(SVDD.get_support_vectors(model))
        add_evaluation_stats!(scores, time_train, time_pred, gamma, C, num_support_vectors)
        return scores
    catch
        return Dict{Symbol, Any}(:eval_status => string(EvaluationException))
    end
end

function train_pwc_model(init_strategy::Dict{Symbol, Any}, data::Array{Float64, 2}, labels::Vector{Symbol})
    c = KDECache(data, init_strategy[:gamma])
    threshold = minimum(kde(c))
    pwc = PWC(c, FixedThresholdStrategy(threshold))
    calculate_threshold!(pwc, labels)
    return pwc
end

function predict_pwc_model(model, test_data)
    return OneClassSampling.predict(model, test_data)
end

function predict_score_pwc_model(model, test_data)
    d_x = kde(model.kde_cache, test_data)
    return OneClassSampling.predict(model, test_data), d_x
end

function evaluate_with_pwc(init_strategy::Dict{Symbol, Any},
                           data::Array{Float64, 2}, labels::Vector{Symbol},
                           test_data::Array{Float64, 2}, test_labels::Vector{Symbol},
                           quality_metrics)
    try
        time_train = @elapsed model = train_pwc_model(init_strategy, data, labels)
        time_pred = @elapsed pred, pred_scores = predict_score_pwc_model(model, test_data)
        scores = evaluate_prediction(quality_metrics, test_labels, pred, pred_scores)
        add_evaluation_stats!(scores, time_train, time_pred, init_strategy[:gamma])
        return scores
    catch
        return Dict{Symbol, Any}(:eval_status => string(EvaluationException))
    end
end

function add_evaluation_stats!(scores, time_train, time_pred, gamma, C=nothing, num_support_vectors=nothing)
    scores[:time_train] = time_train
    scores[:time_pred] = time_pred
    scores[:eval_status] = :success
    scores[:gamma] = gamma
    if C !== nothing
        scores[:C] = C
    end
    if num_support_vectors !== nothing
        scores[:num_support_vectors] = num_support_vectors
    end
    return nothing
end

function evaluate_prediction(quality_metrics, labels, pred, pred_scores)
    scores = Dict{Symbol, Any}()
    cm = ConfusionMatrix(pred, labels)
    for (metric_name, metric) in quality_metrics
        if occursin("auc", string(metric_name))
            m = roc_auc(pred_scores, labels)
            scores[:auc] = m
            for k in metric
                auc_fpr = roc_auc(pred_scores, labels, fpr = k)
                auc_fpr_normalized = roc_auc(pred_scores, labels, fpr = k, normalize = true)
                scores[Symbol("auc_fpr_$(k)")] = auc_fpr
                scores[Symbol("auc_fpr_normalized_$(k)")] = auc_fpr_normalized
            end
        else
            m = metric(cm)
            scores[metric_name] = m
        end
    end
    scores[:tp] = cm.tp
    scores[:fp] = cm.fp
    scores[:tn] = cm.tn
    scores[:fn] = cm.fn
    return scores
end

struct EmptySampleException <: Exception end
struct EvaluationException <: Exception end
struct TooLargeSampleException <: Exception
    n::Int
end

function evaluate(sampler::Sampler, models::Dict{Symbol, Any},
                  sample_mask::BitArray{1},
                  train_data::Array{Float64, 2}, train_labels::Vector{Symbol},
                  test_data::Array{Float64, 2}, test_labels::Vector{Symbol},
                  quality_metrics)
    data, labels = train_data[:, sample_mask], train_labels[sample_mask]
    if length(labels) <= 0
        throw(EmptySampleException())
    end
    scores = Dict{Symbol, Any}()
    for (name, init) in models[:svdd][:init_strategies]
        s = evaluate_with_svdd(eval(models[:svdd][:type]), init, models[:svdd][:solver], data, labels,
                               test_data, test_labels, quality_metrics)
        scores = push!(scores, name => s)
    end
    if :pwc in keys(models)
        s = evaluate_with_pwc(models[:pwc], data, labels,
                              test_data, test_labels, quality_metrics)
        push!(scores, :pwc => s)
    end
    scores_specials = evaluate_specials(sampler, models, data, labels,
                                        test_data, test_labels, quality_metrics)
    if scores_specials !== nothing
        scores = merge(scores, scores_specials)
    end
    return scores
end

function evaluate_specials(sampler::Sampler, models::Dict{Symbol, Any},
                           data::Array{Float64, 2}, labels::Vector{Symbol},
                           test_data::Array{Float64, 2}, test_labels::Vector{Symbol},
                           quality_metrics)
    return nothing
end

function evaluate_specials(sampler::Union{HDS, RAPID}, models::Dict{Symbol, Any},
                           data::Array{Float64, 2}, labels::Vector{Symbol},
                           test_data::Array{Float64, 2}, test_labels::Vector{Symbol},
                           quality_metrics)
    pwc = deepcopy(models[:pwc])
    pwc[:threshold] = sampler.threshold
    s = evaluate_with_pwc(pwc, data, labels,
                          test_data, test_labels, quality_metrics)
    return Dict{Symbol, Any}(:pwc_retuned => s)
end
