# One-Class Sampling Evaluation
_Scripts and notebooks to benchmark one-class sampling strategies._

This repository contains scripts and notebooks to reproduce the experiments and analyses of the paper

> Adrian Englhardt, Holger Trittenbach, Daniel Kottke, Bernhard Sick, Klemens Böhm, "Efficient SVDD sampling with approximation guarantees for the decision boundary", Machine Learning (2022), DOI: [10.1007/s10994-022-06149-0](https://doi.org/10.1007/s10994-022-06149-0).


For more information about this research project, see also the one-class sampling [project website](https://www.ipd.kit.edu/ocs/).

The analysis and main results of the experiments can be found under notebooks:

* `example_intro.ipynb`: Figure 1
* `example.ipynb`: Figure 4
* `eval_synthetic.ipynb`: Figure 5
* `eval_dami.ipynb`: Figure 6 and Table 2

To execute the notebooks, make sure you follow the [setup](#setup), and download the [raw results](https://www.ipd.kit.edu/ocs/output.zip) into `data/output/`.

## Prerequisites

The experiments are implemented in [Julia](https://julialang.org/), some of the evaluation notebooks are written in python.
This repository contains code to setup the experiments, to execute them, and to analyze the results.
The one-class classifiers and some other helper methods are implemented in two separate Julia packages: [SVDD.jl](https://github.com/englhardt/SVDD.jl) and [OneClassActiveLearning.jl](https://github.com/englhardt/OneClassActiveLearning.jl).
The one-class sampling strategies are implemented in [OneClassSampling.jl](https://github.com/englhardt/OneClassSampling.jl).

### Setup

Just clone the repo.
```bash
$ git clone https://github.com/englhardt/ocs-evaluation.git
```
* Experiments require Julia 1.3.1, requirements are defined in `Manifest.toml`. To instantiate, start julia in the `ocs-evaluation` directory with `julia --project` and run `julia> ]instantiate`. See [Julia documentation](https://docs.julialang.org/en/v1.3/stdlib/Pkg/#Using-someone-else's-project-1) for general information on how to setup this project.
* Notebooks require
  * Julia 1.3.1 (dependencies are already installed in the previous step)
  * Python 3.8 and `pipenv`. Run `pipenv install` to install all dependencies

## Repo Overview

* `data`
  * `input`
    * `raw`: contains unprocessed data set collections `literature` and `semantic` downloaded from the [DAMI](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/) repository
    * `dami`: output directory of _preprocess_data.jl_
    * `synthetic`: output directory of _generate_synthetic_data.jl_
  * `output`: output directory of experiments; _generate_experiments.jl_ creates the folder structure and experiments; _run_experiments.jl_ writes results and log files
* `notebooks`: jupyter notebooks to analyze experimental results
  * `eval_dami.ipynb`: Figure 6 and Table 2
  * `eval_synthetic.ipynb`: Figure 5
  * `example_intro.ipynb`: Figure 1
  * `example.ipynb`: Figure 4
* `scripts`
  * `config`: configuration files for experiments
    * `config.jl`: high-level configuration for DAMI experiments, e.g., for number of workers
    * `config_syn.jl`: high-level configuration for synthetic data experiments, e.g., for number of workers
    * `config_dami_large.jl`: experiment config for large DAMI data sets
    * `config_dami.jl`: experiment config for small DAMI data sets
    * `config_dami_baseline_gt.jl`: experiment config for the ground-truth baseline
    * `config_dami_baseline_prefiltering.jl`: experiment config for the prefiltering baseline
    * `config_dami_baseline_rand.jl`: experiment config for the random sample baseline
    * `config_dami_large_outperc.jl`: experiment config for varying the outlier percentage on DAMI data sets
    * `config_dami_outperc.jl`: experiment config for varying the outlier percentage on small DAMI data sets
    * `config_synthetic.jl`: experiment config for synthetic data
    * `config_precompute_parameters.jl`: experiment config to precompute classifier hyperparameters for DAMI data
    * `config_precompute_parameters_gt.jl`: experiment config to precompute classifier hyperparameters for DAMI data with ground truth
    * `config_precompute_parameters_syn.jl`: experiment config to precompute classifier hyperparameters for synthetic data
    * `config_warmup.jl`: experiment config for precomputation warmup experiments
  * `util/setup_workers.jl`: utility script to setup multiple workers, see [Infrastructure and Parallelization](#infrastructure-and-parallelization)
  * `util/evaluate.jl`: utility script to setup evaluate SVDD classifier on samples
  * `generate_experiments.jl`: generate experiments for one type of query strategy, e.g. DAMI
  * `generate_synthetic_data.jl`: generate synthetic data sets
  * `precompute_parameters.jl`: precompute classifier hyperparameters
  * `precompute_parameters_gt.jl`: precompute classifier hyperparameters with ground truth
  * `preprocess_data.jl`: preprocess DAMI data
  * `run_experiments.jl`: executes experiments

## Reproduce Experiments

Here, we specify how to reproduce our experiments after running the steps specified in (Setup)[#setup]

1. Experiment execution

To manually rerun all our experiments we provide two scripts `run.sh` for the DAMI experiments and `run_syn.sh` for the experiments on synthetic data. Since experiment execution takes several days on modern machines, we provide the raw results as a [download](https://www.ipd.kit.edu/ocs/output.zip). One can then skip the experiment execution and head straight to Step 2. The downloaded raw results must be extracted into `data/output/`, e.g., `data/output/dami`

To reproduce the DAMI experiments, download [semantic.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/semantic.tar.gz) and [literature.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/literature.tar.gz) containing the .arff files from the DAMI benchmark repository and extract into `data/input/raw/.../<data set>` (e.g. `data/input/raw/literature/ALOI/` or `data/input/raw/semantic/Annthyroid`).

2. Experiment evaluation

To analyze the results run the jupyter notebooks in the notebooks directory. Run the following to produce the figures and tables in the experiment section of the paper:

```bash
pipenv run eval
pipenv run eval_syn
```

## Infrastructure and Parallelization

Experiment execution can be parallelized over several workers. In general, one can use any [ClusterManager](https://github.com/JuliaParallel/ClusterManagers.jl). In this case, the node that executes `run_experiments.jl` is the driver node. The driver node loads the `experiments.jser`, and initiates a function call for each experiment on one of the workers via `pmap`. Edit `scripts/config/config_syn.jl` and `scripts/config/config.jl` to add remote machines and workers.

## Authors
This package is developed and maintained by [Adrian Englhardt](https://github.com/englhardt/)
