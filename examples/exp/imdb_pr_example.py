import numpy as np
import pandas as pd
from datasets import load_dataset

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_PR


ds = load_dataset('stanfordnlp/imdb', split='test')
np.random.seed(42)
idxs = np.random.choice(len(ds), 400, replace=False)
texts = [ds[int(i)]['text'][:2000] for i in idxs]  # truncate long reviews
labels = [1 if ds[int(i)]['label'] == 1 else 0 for i in idxs]  # 1 = positive sentiment
print(f"Using {len(texts)} IMDB reviews ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

task = '''
I will give you a movie review. Your task is to determine if the sentiment of this review is positive.

- True if the review expresses a positive sentiment
- False otherwise

Here is the review: {}

You must respond with ONLY True or False:
'''

proxy = OpenAIProxy(task, model='gpt-4o-mini', is_binary=True)
oracle = OpenAIOracle(task, model='gpt-4o', is_binary=True)

target = 0.9
delta = 0.1
bargain = BARGAIN_PR(proxy, oracle, delta=delta, target=target, W=50, seed=0)
est_positive_idxs = bargain.process(texts)
oracle_calls = oracle.get_number_preds()
print(f"\nOracle calls during BARGAIN_PR: {oracle_calls}")
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
print(f"  Oracle calls during process: {oracle_calls}/{len(texts)}")
