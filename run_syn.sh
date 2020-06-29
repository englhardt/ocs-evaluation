#!/bin/bash

SCENARIO="synthetic"
mkdir -p logs/$SCENARIO
julia --project scripts/generate_synthetic_data.jl 2>&1 | tee logs/$SCENARIO/generate_data.log
julia --project scripts/precompute_parameters.jl $(pwd)/scripts/config/config_precompute_parameters_syn.jl 2>&1 | tee logs/$SCENARIO/precompute_params.log
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate_experiments.log
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_warmup.jl | tee logs/generate_warump.log
julia --project scripts/run_experiments.jl $(pwd)/scripts/config/config_syn.jl 2>&1 | tee logs/$SCENARIO/run.log
