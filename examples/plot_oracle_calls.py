import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_PR

targets = [0.75, 0.85, 0.90, 0.95, 0.99]
delta = 0.1

datasets_config = {
    'AG News (400)': {
        'load': lambda: load_dataset('fancyzhx/ag_news', split='test'),
        'n': 400,
        'text_fn': lambda ds, i: ds[int(i)]['text'],
        'task': '''
I will give you a news article. Your task is to determine if this article is about Sports.

- True if it is about Sports
- False otherwise

Here is the article: {}

You must respond with ONLY True or False:
'''
    },
    'IMDB (400)': {
        'load': lambda: load_dataset('stanfordnlp/imdb', split='test'),
        'n': 400,
        'text_fn': lambda ds, i: ds[int(i)]['text'][:2000],
        'task': '''
I will give you a movie review. Your task is to determine if the sentiment of this review is positive.

- True if the review expresses a positive sentiment
- False otherwise

Here is the review: {}

You must respond with ONLY True or False:
'''
    },
    'Yelp (400)': {
        'load': lambda: load_dataset('Yelp/yelp_review_full', split='test'),
        'n': 400,
        'text_fn': lambda ds, i: ds[int(i)]['text'][:2000],
        'task': '''
I will give you a Yelp review. Your task is to determine if the reviewer's overall sentiment is positive (4 or 5 stars).

- True if the review is positive (4 or 5 stars)
- False otherwise (1, 2, or 3 stars)

Here is the review: {}

You must respond with ONLY True or False:
'''
    },
    'DBpedia (1000)': {
        'load': lambda: load_dataset('fancyzhx/dbpedia_14', split='test'),
        'n': 1000,
        'text_fn': lambda ds, i: ds[int(i)]['content'][:2000],
        'task': '''
I will give you a short text description. Your task is to determine if this text is about a Company or Corporation.

- True if it is about a company or corporation
- False otherwise

Here is the text: {}

You must respond with ONLY True or False:
'''
    },
}

results = {}

for ds_name, cfg in datasets_config.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")

    ds = cfg['load']()
    np.random.seed(42)
    idxs = np.random.choice(len(ds), cfg['n'], replace=False)
    texts = [cfg['text_fn'](ds, i) for i in idxs]

    proxy = OpenAIProxy(cfg['task'], model='gpt-4o-mini', is_binary=True)
    oracle = OpenAIOracle(cfg['task'], model='gpt-4o', is_binary=True)

    # Pre-warm proxy cache
    print("Warming proxy cache...")
    data_idxs = np.arange(len(texts))
    proxy.get_preds_and_scores(data_idxs, np.array(texts))
    saved_proxy_cache = dict(proxy.preds_dict)
    print(f"Proxy cache warmed: {len(saved_proxy_cache)} entries")

    oracle_calls_list = []
    precisions = []
    recalls = []

    for target in targets:
        print(f"\n  target={target}")
        proxy.preds_dict = dict(saved_proxy_cache)
        proxy.reset = lambda: None
        oracle.reset()

        bargain = BARGAIN_PR(proxy, oracle, delta=delta, target=target, W=50, seed=0, verbose=False)
        est_positive_idxs = bargain.process(texts)
        oc = oracle.get_number_preds()
        oracle_calls_list.append(oc)
        print(f"    oracle calls: {oc}/{len(texts)}, returned: {len(est_positive_idxs)}")

    results[ds_name] = oracle_calls_list
    # Restore proxy.reset
    proxy.reset = lambda self=proxy: setattr(self, 'preds_dict', {}) or None

fig, ax = plt.subplots(figsize=(8, 5))
markers = ['o', 's', '^', 'D']
for i, (ds_name, calls) in enumerate(results.items()):
    n = int(ds_name.split('(')[1].rstrip(')'))
    fractions = [c / n for c in calls]
    ax.plot(targets, fractions, marker=markers[i], label=ds_name, linewidth=2, markersize=8)

ax.set_xlabel('Target (precision & recall)', fontsize=12)
ax.set_ylabel('Fraction of oracle calls', fontsize=12)
ax.set_title('BARGAIN-PR: Oracle calls vs Target', fontsize=14)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_xticks(targets)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/BARGAIN/examples/oracle_calls_plot.png', dpi=150)
print(f"\nPlot saved to examples/oracle_calls_plot.png")
