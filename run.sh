#!/bin/bash

SCENARIO="dami"
mkdir -p logs/$SCENARIO
julia --project scripts/preprocess_data.jl $(pwd)/scripts/config/config.jl
julia --project scripts/precompute_parameters.jl $(pwd)/scripts/config/config_precompute_parameters.jl
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
julia --project scripts/run_experiments.jl $(pwd)/scripts/config/config.jl 2>&1 | tee logs/$SCENARIO/run.log

SCENARIO="dami_large"
mkdir -p logs/$SCENARIO
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
julia --project scripts/run_experiments.jl $(pwd)/scripts/config/config.jl 2>&1 | tee logs/$SCENARIO/run.log

SCENARIO="dami_baseline_rand"
mkdir -p logs/$SCENARIO
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
SCENARIO="dami_baseline_prefiltering"
mkdir -p logs/$SCENARIO
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
SCENARIO="dami_baseline_gt"
mkdir -p logs/$SCENARIO
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
SCENARIO="dami_baseline"
mkdir -p logs/$SCENARIO
julia --project scripts/run_experiments.jl $(pwd)/scripts/config/config.jl 2>&1 | tee logs/$SCENARIO/run.log

SCENARIO="dami_outperc"
mkdir -p logs/$SCENARIO
julia --project scripts/preprocess_data.jl $(pwd)/scripts/config/config.jl
julia --project scripts/precompute_parameters.jl $(pwd)/scripts/config/config_precompute_parameters.jl
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
julia --project scripts/run_experiments.jl $(pwd)/scripts/config/config.jl 2>&1 | tee logs/$SCENARIO/run.log

SCENARIO="dami_large_outperc"
mkdir -p logs/$SCENARIO
julia --project scripts/preprocess_data.jl $(pwd)/scripts/config/config.jl
julia --project scripts/precompute_parameters.jl $(pwd)/scripts/config/config_precompute_parameters.jl
julia --project scripts/generate_experiments.jl $(pwd)/scripts/config/config_$SCENARIO.jl 2>&1 | tee logs/$SCENARIO/generate.log
julia --project scripts/run_experiments.jl $(pwd)/scripts/config/config.jl 2>&1 | tee logs/$SCENARIO/run.log
