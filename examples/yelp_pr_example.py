import numpy as np
import pandas as pd
from datasets import load_dataset

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_PR


# Yelp polarity: binary sentiment (1 = positive, 2 = positive in the dataset)
ds = load_dataset('Yelp/yelp_review_full', split='test')
np.random.seed(42)
idxs = np.random.choice(len(ds), 400, replace=False)
texts = [ds[int(i)]['text'][:2000] for i in idxs]
labels = [1 if ds[int(i)]['label'] >= 3 else 0 for i in idxs]  # 4-5 stars = positive
print(f"Using {len(texts)} Yelp reviews ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

task = '''
I will give you a Yelp review. Your task is to determine if the reviewer's overall sentiment is positive (4 or 5 stars).

- True if the review is positive (4 or 5 stars)
- False otherwise (1, 2, or 3 stars)

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
