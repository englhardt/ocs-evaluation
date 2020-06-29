import os
import json
import numpy as np
import pandas as pd
import pickle
import re
from pandas.io.json import json_normalize
from tqdm import tqdm

pretty_names = {
    'model_init_strategy': {
        "SimpleCombinedStrategy(RuleOfThumbScott(), FixedCStrategy(1))": 'RoT_C1',
        "SimpleCombinedStrategy(RuleOfThumbScott(), BoundedTaxErrorEstimate(0.05, 0.02, 0.98))": 'RoT_CTax',
        "WangCombinedInitializationStrategy(OptimizerFactory(Gurobi.Optimizer, (), Base.Iterators.Pairs(:OutputFlag => 0,:Threads => 1)), [0.01, 0.012067926406393288, 0.014563484775012436, 0.017575106248547922, 0.021209508879201904, 0.025595479226995357, 0.030888435964774818, 0.0372759372031494, 0.044984326689694466, 0.054286754393238594, 0.0655128556859551, 0.07906043210907701, 0.09540954763499938, 0.11513953993264472, 0.13894954943731377, 0.16768329368110083, 0.20235896477251572, 0.2442053094548651, 0.29470517025518106, 0.35564803062231287, 0.4291934260128778, 0.5179474679231212, 0.6250551925273973, 0.7543120063354617, 0.9102981779915219, 1.0985411419875581, 1.3257113655901092, 1.599858719606058, 1.93069772888325, 2.329951810515372, 2.8117686979742307, 3.393221771895329, 4.094915062380426, 4.941713361323834, 5.963623316594643, 7.196856730011519, 8.685113737513527, 10.481131341546858, 12.648552168552959, 15.264179671752334, 18.420699693267164, 22.229964825261945, 26.826957952797258, 32.374575428176435, 39.06939937054617, 47.14866363457394, 56.89866029018296, 68.66488450043002, 82.86427728546842, 100.0], BoundedTaxErrorEstimate(0.05, 0.02, 0.98))": 'Wang_Tax'
    },
    'threshold_strategy': {
        'MaxDensityThresholdStrategy': 'MaxDens',
        'OutlierPercentageThresholdStrategy': 'OutPerc',
        'GroundTruthThresholdStrategy': 'GroundT',
        'nothing': 'none',
    },
    'ss': {
        'RandomRatioSampler': 'Rand',
        'HighestDensitySampler': 'HDS'
    }
}

def process_sampling_strategy(raw):
    s = raw.replace('(', ',').replace(')', ',').split(",")
    x = {'ss': pretty_names['ss'].get(s[0], s[0]), 'sst': pretty_names['ss'].get(s[0], s[0]), 'ss_threshold_strat': np.nan, 'ss_n': np.nan, 'ss_k': np.nan, 'ss_init': np.nan,
         'ss_num_chunks': np.nan, 'ss_chunk_size': np.nan, 'ss_num_levels': np.nan}
    if 'PreFiltering' in raw and 'HSC' in raw:
        sub_sampler = raw.split('), OutlierPercentageThresholdStrategy')[0].replace('PriorPruningWrapper(', '').replace('PreFilteringWrapper(', '')
        x = process_sampling_strategy(sub_sampler)
        x['ss'] = f'PP_{x["ss"]}'
        x['sst'] = f'PP_{x["sst"]}'
    elif 'PreFiltering' in raw:
        split2 = re.split(r'(\w+Wrapper)\((.*), (\w+ThresholdStrategy)\((.*)\), \d+.\d+, KDECache', raw, re.I)
        sub_sampler = split2[2]
        x = process_sampling_strategy(sub_sampler)
        x['ss'] = f'PP_{x["ss"]}'
        x['sst'] = f'PP_{x["sst"]}'
        x['ss_threshold_strat'] = f"{pretty_names['threshold_strategy'][split2[3]]}({split2[4]})"
    elif 'HighestDensitySampler' in raw or 'HDS' in raw and not 'HDSM' in raw:
        threshold_strat = pretty_names['threshold_strategy'].get(s[2].strip(), s[2].strip())
        x['ss_threshold_strat'] = f"{threshold_strat}({s[3]})"
        x['ss_eps'] = float(s[1])
        x['ss'] = f"{pretty_names['ss'].get(s[0], s[0])}({s[1]}, {x['ss_threshold_strat']})"
    elif 'RAPID' in raw:
        x['ss_init'] = s[-2].replace(':', '').strip()
        threshold_strat = pretty_names['threshold_strategy'].get(s[1].strip(), s[1])
        x['ss_threshold_strat'] = f"{threshold_strat}({s[2]})"
        x['ss'] = f"{pretty_names['ss'].get(s[0], s[0])}({x['ss_threshold_strat']})"
    elif any(x in raw for x in ['DAEDS', 'DBSRSVDD', 'HSR', 'NDPSR']):
        x['ss'] = raw
        x['ss_eps'] = float(s[2])
        x['ss_k'] = int(s[1])
    elif 'KFNCBD' in raw:
        x['ss_k'] = int(s[1])
        if len(s) > 8:
            threshold_strat = pretty_names['threshold_strategy'].get(s[2].strip(), s[2].strip())
            x['ss_threshold_strat'] = f"{threshold_strat}({s[3]})"
            x['ss'] = f"{pretty_names['ss'].get(s[0], s[0])}({s[1]}, {x['ss_threshold_strat']})"
        else:
            x['ss'] = raw
            x['ss_eps'] = float(s[2])
    elif 'BPS' in raw:
        x['ss_eps'] = float(s[2])
        x['ss_k'] = int(s[1])
        x['ss'] = f"{s[0]}({x['ss_eps']})"
    elif 'HSC' in raw:
        x['ss_k'] = int(s[1])
        threshold_strat = s[3].strip()
        threshold_strat = pretty_names['threshold_strategy'].get(threshold_strat, threshold_strat)
        x['ss_threshold_strat'] = f"{threshold_strat}({s[4]})"
        x['ss'] = f"{s[0]}({x['ss_k']},{s[2]}, {x['ss_threshold_strat']})"
    elif 'KMSVDD' in raw:
        x['ss_k'] = int(s[1])
        x['ss'] = f"{s[0]}({x['ss_k']})"
    elif 'CSSVDD' in raw:
        x['ss_n'] =  int(s[1])
        x['ss_k'] = int(s[2])
        x['ss_eps'] = float(s[4])
        x['ss'] = f"{s[0]}({x['ss_n']}, {x['ss_k']}, {x['ss_eps']})"
    elif 'RSSVDD' in raw:
        x['ss_k'] = int(s[1])
        x['ss_n'] =  int(s[2])
        x['ss'] = f"{s[0]}({x['ss_k']}, {x['ss_n']})"
    elif 'FBPE' in raw:
        x['ss_n'] = int(s[1])
        x['ss'] = raw
    elif 'IESRSVDD' in raw:
        x['ss_eps'] = float(s[-2])
        x['ss'] = f"{s[0]}({x['ss_eps']})"
    elif "RandomRatioSampler" in raw:
        x['ss'] = f"{pretty_names['ss'].get(s[0], s[0])}({s[1]})"
    else:
        x['ss'] = raw
    return x


def load_raw_results(p):
    results = pd.DataFrame()
    raw_files = []
    for subdir, dirs, files in os.walk(os.path.join(p, "results")):
        raw_files += [os.path.join(subdir, f) for f in files]
    for f in tqdm(raw_files):
        with open(f) as res_file:
            try:
                res = json.load(res_file)
                results = results.append(json_normalize(res), sort=True)
            except json.JSONDecodeError as e:
                print(f"Cannot parse {f}")
                print(e)
    return results

def pretty_df(results):
    results = results.set_index('hash')
    ss_new_cols = pd.DataFrame(results.sampling_strategy.apply(process_sampling_strategy).tolist(), index=results.index)
    results = results.merge(ss_new_cols, left_index=True, right_index=True)
    for c in ['data_stats', 'result']:
        results.columns = [x if c not in x else x.replace(f'{c}.', '') for x in results.columns]
    results.columns = [x.replace('.', '_') for x in results]
    results['scenario'] = results.log_dir.values[0].split('/')[-2]
    results['data_file'] = [x.split('/')[-1] for x in results.data_file.values]
    results['output_file'] = [x.split('/')[-1] for x in results.output_file.values]
    results['log_dir'] = results.log_dir.values[0].split('/')[-1]
    results['sample_ratio'] = results['sample_size'] / results['n_observations']
    return results

PKL_FILE = 'results.pkl'
CSV_FILE = 'results.csv'
def load_results(exp_paths, data_output_root='../data/output', force_pkl_redo=False, force_csv_redo=False):
    rs = []
    for p in exp_paths:
        if not os.path.isdir(os.path.join(data_output_root, p)):
            print(f"Cannot find '{os.path.join(data_output_root, p)}'.")
            return
        p_pkl_file = os.path.join(data_output_root, p, PKL_FILE)
        p_csv_file = os.path.join(data_output_root, p, CSV_FILE)
        if not os.path.isfile(p_csv_file) or force_pkl_redo or force_csv_redo:
            if not os.path.isfile(p_pkl_file) or force_pkl_redo:
                r = load_raw_results(os.path.join(data_output_root, p))
                with open(p_pkl_file, 'wb') as f:
                    pickle.dump(r, f)
            else:
                with open(p_pkl_file, 'rb') as f:
                    r = pickle.load(f)
            r_pretty = pretty_df(r)
            r_pretty.to_csv(p_csv_file)
        else:
            r_pretty = pd.read_csv(p_csv_file, index_col='hash')
        rs.append(r_pretty)

    r = pd.concat(rs, sort=False)
    return r


def highlight_max(data, color='gray'):
    attr = f'background-color: {color}'
    if data.ndim == 1:
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

def highlight_min(data, color='gray'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
