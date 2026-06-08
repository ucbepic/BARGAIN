import numpy as np
import pandas as pd
from datasets import load_dataset

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_PR


# DBpedia has 14 classes; we frame as binary: "Company" (class 0) vs everything else
ds = load_dataset('fancyzhx/dbpedia_14', split='test')
np.random.seed(42)
idxs = np.random.choice(len(ds), 400, replace=False)
texts = [ds[int(i)]['content'][:2000] for i in idxs]
labels = [1 if ds[int(i)]['label'] == 0 else 0 for i in idxs]  # Company = positive
print(f"Using {len(texts)} DBpedia articles ({sum(labels)} Company, {len(labels)-sum(labels)} other)")

task = '''
I will give you a short text description. Your task is to determine if this text is about a Company or Corporation.

- True if it is about a company or corporation
- False otherwise

Here is the text: {}

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

gt_labels = np.array(labels)
if len(est_positive_idxs) == 0:
    est_prec = 1.0
else:
    est_prec = gt_labels[est_positive_idxs].mean()

total_pos = gt_labels.sum()
est_rec = gt_labels[est_positive_idxs].sum() / total_pos if total_pos > 0 else 1.0

print(f"\nResults (target={target}, delta={delta}):")
print(f"  Precision: {est_prec:.3f}")
print(f"  Recall:    {est_rec:.3f}")
print(f"  Oracle calls: {oracle_calls}/{len(texts)}")
