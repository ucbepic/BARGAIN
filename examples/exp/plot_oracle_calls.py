import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_dataset(ds_name, cfg):
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
    print(f"[{ds_name}] Proxy done ({len(saved_proxy_cache)} entries)")

    print(f"[{ds_name}] Warming oracle cache...")
    oracle_labels = oracle.get_pred(texts_arr, data_idxs)
    n_pos = int(oracle_labels.sum())
    print(f"[{ds_name}] Oracle done ({n_pos} positive, {len(texts) - n_pos} negative)")

    oracle_calls_list = []
    precisions = []
    recalls = []

    for target in targets:
        proxy.preds_dict = dict(saved_proxy_cache)
        proxy.reset = lambda: None
        oracle.reset()

        bargain = BARGAIN_PR(proxy, oracle, delta=delta, target=target, W=50, seed=0, verbose=False)
        est_positive_idxs = bargain.process(texts)
        oc = oracle.get_number_preds()
        oracle_calls_list.append(oc)

        if len(est_positive_idxs) == 0:
            prec = 1.0
        else:
            prec = float(oracle_labels[est_positive_idxs].mean())

        total_pos = oracle_labels.sum()
        rec = float(oracle_labels[est_positive_idxs].sum() / total_pos) if total_pos > 0 else 1.0

        precisions.append(prec)
        recalls.append(rec)
        print(f"[{ds_name}] target={target}: oracle={oc}/{len(texts)}, prec={prec:.3f}, rec={rec:.3f}")

    return ds_name, {
        'oracle_calls': oracle_calls_list,
        'precisions': precisions,
        'recalls': recalls,
        'n': cfg['n'],
    }


results = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_dataset, name, cfg): name for name, cfg in datasets_config.items()}
    for future in as_completed(futures):
        ds_name, result = future.result()
        results[ds_name] = result
        print(f"[{ds_name}] COMPLETE")

# Maintain consistent order for plotting
plot_order = [name for name in datasets_config if name in results]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for ds_name in plot_order:
    r = results[ds_name]
    fractions = [c / r['n'] for c in r['oracle_calls']]
    ax.plot(targets, fractions, label=ds_name, linewidth=2)
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Fraction of oracle calls', fontsize=12)
ax.set_title('Oracle Calls', fontsize=14)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xticks(targets)
ax.grid(True, alpha=0.3)

ax = axes[1]
for ds_name in plot_order:
    r = results[ds_name]
    ax.plot(targets, r['precisions'], label=ds_name, linewidth=2)
ax.plot(targets, targets, 'k--', alpha=0.4, label='y=target')
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Precision (oracle)', fontsize=12)
ax.set_title('Precision vs Target', fontsize=14)
ax.legend(fontsize=9)
ax.set_ylim(0.6, 1.05)
ax.set_xticks(targets)
ax.grid(True, alpha=0.3)

ax = axes[2]
for ds_name in plot_order:
    r = results[ds_name]
    ax.plot(targets, r['recalls'], label=ds_name, linewidth=2)
ax.plot(targets, targets, 'k--', alpha=0.4, label='y=target')
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Recall (oracle)', fontsize=12)
ax.set_title('Recall vs Target', fontsize=14)
ax.legend(fontsize=9)
ax.set_ylim(0.6, 1.05)
ax.set_xticks(targets)
ax.grid(True, alpha=0.3)

plt.suptitle('BARGAIN-PR: Performance across datasets (delta=0.1)', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('/home/user/BARGAIN/examples/exp/oracle_calls_plot.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to examples/exp/oracle_calls_plot.png")
