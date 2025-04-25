import numpy as np
import pandas as pd
from tqdm import tqdm

from PRISM import PRISM_R
from PreComputedModels import PrecomputedOracle, PrecomputedProxy

def precision(indexes, labels, pred_indexes):
    if len(pred_indexes) == 0:
        return 1
    df = pd.DataFrame.from_dict({'indexes':indexes, 'labels':labels}).set_index('indexes')
    return df.loc[pred_indexes, 'labels'].mean()

def recall(indexes, labels, pred_indexes):
    df = pd.DataFrame.from_dict({'indexes':indexes, 'labels':labels}).set_index('indexes')
    no_pos =(df['labels']==1).sum()
    no_pos_pred = df.loc[pred_indexes, 'labels'].sum()
    return no_pos_pred/no_pos

target = 0.9
delta = 0.1
k = 400
no_runs = 50

res = {'data':[], 'precision':[], 'recall':[], 'seed':[], 'method':[], 'meets_target':[],'orcale_usage':[] }
for data in ['reviews', 'court_opinion', 'screenplay', 'wiki_talk', 'tacred', 'imagenet', 'onto', 'jackson']:
    df = pd.read_csv(f'data/{data}.csv')
    print(data)
    for seed in tqdm(range(no_runs)):
        oracle = PrecomputedOracle(df['id'].to_numpy(), df['label'].to_numpy())
        preds = df['proxy_score'].to_numpy()>0.5
        scores = df['proxy_score'].to_numpy()
        scores = preds*scores+(1-scores)*(1-preds)
        proxy = PrecomputedProxy(df['id'].to_numpy(), preds, scores)
        prism = PRISM_R(df['id'].to_numpy(), proxy, oracle, delta, target, k, seed=seed)
        est_positve = prism.process()

        curr_recall = recall(df['id'].to_numpy(), df['label'].to_numpy(), est_positve)
        res['precision'].append(precision(df['id'].to_numpy(), df['label'].to_numpy(), est_positve))
        res['recall'].append(curr_recall)
        res['orcale_usage'].append(oracle.get_number_preds())
        res['method'].append("PRISM_R")
        res['meets_target'].append(curr_recall>=target)
        res['seed'].append(seed)
        res['data'].append(data)

df_res = pd.DataFrame.from_dict(res)
print(df_res.groupby(['method', 'data']).mean())