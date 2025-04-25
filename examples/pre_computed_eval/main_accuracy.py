import numpy as np
import pandas as pd
from tqdm import tqdm

from PRISM import PRISM_A 
from PreComputedModels import PrecomputedOracle, PrecomputedProxy

target = 0.9
delta = 0.1
no_runs = 100

res = {'data':[], 'accs':[], 'proxy_usage':[], 'seed':[], 'method':[], 'meets_target':[]}
for data in ['reviews', 'court_opinion', 'screenplay', 'wiki_talk', 'tacred', 'imagenet', 'onto', 'jackson']:
#for data in ['reviews']:
    df = pd.read_csv(f'data/{data}.csv')
    df['proxy_pred'] = df['proxy_score']>=0.5
    df['proxy_score'] = df['proxy_score']*df['proxy_pred']+(1-df['proxy_score'])*(1-df['proxy_pred'])
    for seed in tqdm(range(no_runs)):
        oracle = PrecomputedOracle(df['id'].to_numpy(), df['label'].to_numpy())
        proxy = PrecomputedProxy(df['id'].to_numpy(), df['proxy_pred'].to_numpy(), df['proxy_score'].to_numpy())

        prism = PRISM_A(df['id'].to_numpy(), proxy, oracle, delta, target, seed=seed, verbose=False)
        output_df = prism.process()
        df_joined = df.set_index('id').join(output_df)
        df_joined['is_correct'] = df_joined['label'] == df_joined['output']
        res['accs'].append(df_joined['is_correct'].mean())
        res['proxy_usage'].append(1-oracle.get_number_preds()/len(df))
        res['method'].append("PRISM_A")
        res['meets_target'].append(df_joined['is_correct'].mean()>=target)
        res['seed'].append(seed)
        res['data'].append(data)


df_res = pd.DataFrame.from_dict(res)
print(df_res.groupby(['method', 'data']).mean())
