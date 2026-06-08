import numpy as np
import pandas as pd

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_PR

def precision(all_idxs, oracle_labels, pred_idxs):
    if len(pred_idxs) == 0:
        return 1.0
    df = pd.DataFrame({'idx': all_idxs, 'label': oracle_labels}).set_index('idx')
    return df.loc[pred_idxs, 'label'].mean()

def recall(all_idxs, oracle_labels, pred_idxs):
    df = pd.DataFrame({'idx': all_idxs, 'label': oracle_labels}).set_index('idx')
    total_pos = (df['label'] == 1).sum()
    if total_pos == 0:
        return 1.0
    found_pos = df.loc[pred_idxs, 'label'].sum()
    return found_pos / total_pos

# Define Data and Task
task = '''
I will give you a Supreme Court opinion.
Your task is to determine if this opinion reverses a lower court's ruling.
Note that the opinion may not be an appeal, but rather a new ruling.

- True if the Supreme Court reverses the lower court ruling
- False otherwise

Here is the opinion: {}

You must respond with ONLY True or False:
'''

df = pd.read_csv('court_opinion.csv')
np.random.seed(42)
df = df.sample(n=200, random_state=42).reset_index(drop=True)
print(f"Using {len(df)} court opinions")

# Define oracle and proxy
proxy = OpenAIProxy(task, model='gpt-4o-mini', is_binary=True)
oracle = OpenAIOracle(task, model='gpt-4o', is_binary=True)

# Run BARGAIN_PR
target = 0.9
delta = 0.1
bargain = BARGAIN_PR(proxy, oracle, delta=delta, target=target, W=50, seed=0)
est_positive_idxs = bargain.process(df['opinion_text'].to_numpy())
oracle_calls_during = oracle.get_number_preds()
print(f"\nOracle calls during BARGAIN_PR: {oracle_calls_during}")
print(f"Returned {len(est_positive_idxs)} positive indices")

# Evaluate: run oracle on ALL records to get ground truth
print("\nRunning oracle on all records for evaluation...")
all_idxs = np.arange(len(df))
all_labels = oracle.get_pred(df['opinion_text'].to_numpy(), all_idxs)

est_prec = precision(all_idxs, all_labels, est_positive_idxs)
est_rec = recall(all_idxs, all_labels, est_positive_idxs)
print(f"\nResults (target={target}, delta={delta}):")
print(f"  Precision: {est_prec:.3f}")
print(f"  Recall:    {est_rec:.3f}")
print(f"  Oracle calls during process: {oracle_calls_during}/{len(df)}")
