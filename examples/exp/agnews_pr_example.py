import numpy as np
import pandas as pd
from datasets import load_dataset

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


# Load AG News and sample 400 records
ds = load_dataset('fancyzhx/ag_news', split='test')
np.random.seed(42)
idxs = np.random.choice(len(ds), 400, replace=False)
texts = [ds[int(i)]['text'] for i in idxs]
labels = [1 if ds[int(i)]['label'] == 1 else 0 for i in idxs]  # Sports = positive
print(f"Using {len(texts)} AG News articles ({sum(labels)} Sports, {len(labels)-sum(labels)} other)")

task = '''
I will give you a news article. Your task is to determine if this article is about Sports.

- True if it is about Sports
- False otherwise

Here is the article: {}

You must respond with ONLY True or False:
'''

proxy = OpenAIProxy(task, model='gpt-4o-mini', is_binary=True)
oracle = OpenAIOracle(task, model='gpt-4o', is_binary=True)

target = 0.9
delta = 0.1
bargain = BARGAIN_PR(proxy, oracle, delta=delta, target=target, W=50, seed=0)
est_positive_idxs = bargain.process(texts)
oracle_calls_during = oracle.get_number_preds()
print(f"\nOracle calls during BARGAIN_PR: {oracle_calls_during}")
print(f"Returned {len(est_positive_idxs)} positive indices")

print("\nRunning oracle on all records for evaluation...")
all_idxs = np.arange(len(texts))
all_labels = oracle.get_pred(np.array(texts), all_idxs)

if len(est_positive_idxs) == 0:
    est_prec = 1.0
else:
    est_prec = all_labels[est_positive_idxs].mean()

total_pos = all_labels.sum()
est_rec = all_labels[est_positive_idxs].sum() / total_pos if total_pos > 0 else 1.0

print(f"\nResults (target={target}, delta={delta}):")
print(f"  Precision (oracle): {est_prec:.3f}")
print(f"  Recall (oracle):    {est_rec:.3f}")
print(f"  Oracle calls during process: {oracle_calls_during}/{len(texts)}")
