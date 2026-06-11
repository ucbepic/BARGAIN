import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from BARGAIN import OpenAIProxy, OpenAIOracle
from BARGAIN import BARGAIN_PR

targets = [0.75, 0.85, 0.90, 0.95, 0.99]
delta = 0.1
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

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


def _cache_key(ds_name):
    return ds_name.replace(' ', '_').replace('(', '').replace(')', '').lower()


def run_dataset(ds_name, cfg):
    cache_file = os.path.join(CACHE_DIR, f'{_cache_key(ds_name)}.pkl')

    if os.path.exists(cache_file):
        print(f"[{ds_name}] Loading from disk cache...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        texts = cached['texts']
        saved_proxy_cache = cached['proxy_cache']
        saved_oracle_cache = cached['oracle_cache']
        print(f"[{ds_name}] Cache loaded")
    else:
        print(f"[{ds_name}] Loading dataset...")
        ds = cfg['load']()
        np.random.seed(42)
        idxs = np.random.choice(len(ds), cfg['n'], replace=False)
        texts = [cfg['text_fn'](ds, i) for i in idxs]
        data_idxs = np.arange(len(texts))
        texts_arr = np.array(texts)

        proxy = OpenAIProxy(cfg['task'], model='gpt-4o-mini', is_binary=True)
        oracle = OpenAIOracle(cfg['task'], model='gpt-4o', is_binary=True)

        print(f"[{ds_name}] Warming proxy cache...")
        proxy.get_preds_and_scores(data_idxs, texts_arr)
        saved_proxy_cache = dict(proxy.preds_dict)
        print(f"[{ds_name}] Proxy done")

        print(f"[{ds_name}] Warming oracle cache...")
        oracle_labels = oracle.get_pred(texts_arr, data_idxs)
        saved_oracle_cache = dict(oracle.preds_dict)
        print(f"[{ds_name}] Oracle done")

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'texts': texts,
                'proxy_cache': saved_proxy_cache,
                'oracle_cache': saved_oracle_cache,
                'oracle_labels': oracle_labels,
            }, f)
        print(f"[{ds_name}] Saved to disk cache")

    proxy = OpenAIProxy(cfg['task'], model='gpt-4o-mini', is_binary=True)
    oracle = OpenAIOracle(cfg['task'], model='gpt-4o', is_binary=True)

    thresholds = []
    for target in targets:
        proxy.preds_dict = dict(saved_proxy_cache)
        proxy.reset = lambda: None
        oracle.preds_dict = dict(saved_oracle_cache)
        oracle.reset = lambda: None
        queried = set()
        _orig_get_pred = type(oracle).get_pred
        def _tracking_get_pred(self_oracle, data_records, indxs=None):
            if indxs is not None:
                for idx in indxs:
                    queried.add(int(idx))
            return _orig_get_pred(self_oracle, data_records, indxs)
        oracle.get_pred = lambda *a, **kw: _tracking_get_pred(oracle, *a, **kw)

        bargain = BARGAIN_PR(proxy, oracle, delta=delta, target=target, W=50, seed=0, verbose=False)
        bargain.process(texts)
        n = bargain.n_
        t_P = bargain.t_P_
        not_queried_accept = len([i for i in range(t_P, n) if i not in queried])
        oracle_frac = len(queried) / n
        accept_frac = not_queried_accept / n
        reject_frac = 1 - oracle_frac - accept_frac
        thresholds.append((reject_frac, oracle_frac, accept_frac))
        print(f"[{ds_name}] target={target}: reject={reject_frac:.3f}, oracle={oracle_frac:.3f}, accept={accept_frac:.3f}")

    print(f"[{ds_name}] COMPLETE")
    return ds_name, thresholds


results = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_dataset, name, cfg): name for name, cfg in datasets_config.items()}
    for future in as_completed(futures):
        ds_name, thresholds = future.result()
        results[ds_name] = thresholds

plot_order = [name for name in datasets_config if name in results]
colors = {'reject': '#e74c3c', 'oracle': '#f39c12', 'accept': '#2ecc71'}

fig, axes = plt.subplots(len(plot_order), len(targets), figsize=(18, 6),
                          gridspec_kw={'hspace': 0.6, 'wspace': 0.3})

for row, ds_name in enumerate(plot_order):
    for col, target in enumerate(targets):
        ax = axes[row, col]
        rej, orc, acc = results[ds_name][col]

        ax.barh(0, rej, left=0, height=0.5, color=colors['reject'], alpha=0.8)
        ax.barh(0, orc, left=rej, height=0.5, color=colors['oracle'], alpha=0.8)
        ax.barh(0, acc, left=rej + orc, height=0.5, color=colors['accept'], alpha=0.8)

        if rej > 0.05:
            ax.text(rej / 2, 0, f'{rej:.0%}', ha='center', va='center', fontsize=8, fontweight='bold')
        if orc > 0.05:
            ax.text(rej + orc / 2, 0, f'{orc:.0%}', ha='center', va='center', fontsize=8, fontweight='bold')
        if acc > 0.05:
            ax.text(rej + orc + acc / 2, 0, f'{acc:.0%}', ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.4)
        ax.set_yticks([])
        if row == 0:
            ax.set_title(f'target={target}', fontsize=10)
        if col == 0:
            ax.set_ylabel(ds_name.split(' (')[0], fontsize=10, rotation=0, labelpad=60, va='center')
        if row == len(plot_order) - 1:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'], fontsize=8)
        else:
            ax.set_xticks([])

legend_patches = [
    mpatches.Patch(color=colors['reject'], alpha=0.8, label='Proxy only (not queried)'),
    mpatches.Patch(color=colors['oracle'], alpha=0.8, label='Oracle queried'),
    mpatches.Patch(color=colors['accept'], alpha=0.8, label='Proxy only (accepted)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle('BARGAIN-PR: Threshold positions (fraction of sorted data)', fontsize=14, y=1.01)
plt.savefig('/home/user/BARGAIN/examples/exp/thresholds_plot.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to examples/exp/thresholds_plot.png")
