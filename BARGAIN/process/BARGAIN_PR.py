import numpy as np
from typing import List

from BARGAIN.sampler.wor_sampler import WoR_Sampler
from BARGAIN.models.AbstractModels import Oracle, Proxy
from BARGAIN.bounds.betting_bounds import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m


class BARGAIN_PR():
    '''
    Class to process a dataset using a cheap proxy or an expensive oracle while guaranteeing
    both desired precision and recall targets are met simultaneously.
    '''
    def __init__(
        self,
        proxy: Proxy,
        oracle: Oracle,
        delta: float = 0.1,
        target: float = 0.9,
        W: int = 50,
        M: int = 20,
        verbose: bool = True,
        seed: int = 0
    ):
        '''
        Args:
            proxy: Proxy model to use
            oracle: Oracle model to use
            delta: Probability of failure, float between 0 and 1
            target: Desired precision and recall target, float between 0 and 1
            W: Sliding window size for precision threshold search
            M: Number of thresholds for BARGAIN fallback precision search
            verbose: Output progress details
            seed: Random seed
        '''
        self.delta = delta
        self.target = target
        self.W = W
        self.M = M
        self.proxy = proxy
        self.oracle = oracle
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

    def _get_labels(self, sorted_indices, data_idxs, data_records):
        sorted_indices = np.array(sorted_indices, dtype=int)
        return self.oracle.get_pred(data_records[sorted_indices], data_idxs[sorted_indices])

    def _find_recall_threshold(self, data_idxs, data_records, n, labeled):
        """
        Find t_R: the rightmost sorted index such that >= target fraction
        of all true positives are at sorted index >= t_R, with confidence delta/2.
        """
        delta_recall = self.delta / 2
        sample_step = 50

        perm = np.random.permutation(n)
        ptr = 0
        pos_sorted_indices = []
        best_t = 0

        while ptr < n:
            batch_end = min(ptr + sample_step, n)
            batch = perm[ptr:batch_end]
            ptr = batch_end

            new_in_batch = np.array([i for i in batch if i not in labeled])
            if len(new_in_batch) > 0:
                labels = self._get_labels(new_in_batch, data_idxs, data_records)
                for idx, lbl in zip(new_in_batch, labels):
                    labeled[idx] = lbl

            for i in batch:
                if labeled[i] == 1:
                    pos_sorted_indices.append(i)

            if len(pos_sorted_indices) < 10:
                continue

            pos_arr = np.array(pos_sorted_indices)
            threshs = np.sort(np.unique(pos_arr))

            found = False
            candidate_t = 0
            for t in threshs:
                obs = (pos_arr >= t).astype(float)
                if np.mean(obs) < self.target:
                    break
                conf_in_ci = test_if_true_mean_is_above_m(
                    obs, self.target, alpha=delta_recall
                )
                if not conf_in_ci:
                    candidate_t = t
                    found = True
                else:
                    break

            if found:
                best_t = candidate_t
                break

        return best_t

    def _sliding_window_precision(self, t_R, data_idxs, data_records, n, labeled):
        """
        Slide a window of size W from t_R rightward. Label points in the window
        and check empirical precision. Returns the left edge of the first window
        where precision >= target, or n if none found.
        """
        W = self.W
        if t_R + W > n:
            return n

        to_label = [i for i in range(t_R, t_R + W) if i not in labeled]
        if to_label:
            labels = self._get_labels(to_label, data_idxs, data_records)
            for idx, lbl in zip(to_label, labels):
                labeled[idx] = lbl

        window_start = t_R
        while window_start + W <= n:
            window_labels = np.array([labeled[i] for i in range(window_start, window_start + W)])
            if np.mean(window_labels) >= self.target:
                return window_start

            window_start += 1
            new_idx = window_start + W - 1
            if new_idx < n and new_idx not in labeled:
                lbl = self._get_labels([new_idx], data_idxs, data_records)
                labeled[new_idx] = lbl[0]

        return n

    def _validate_precision(self, t_P, data_idxs, data_records, n, labeled, delta_val):
        """
        Validate that precision >= target in [t_P, n) using adaptive sampling
        with betting bounds. Returns True if validation passes.
        """
        region_size = n - t_P
        if region_size == 0:
            return True

        sample_step = 10
        perm = np.random.permutation(np.arange(t_P, n))
        ptr = 0
        all_obs = []

        while ptr < len(perm):
            batch_end = min(ptr + sample_step, len(perm))
            batch = perm[ptr:batch_end]
            ptr = batch_end

            new_in_batch = np.array([i for i in batch if i not in labeled])
            if len(new_in_batch) > 0:
                labels = self._get_labels(new_in_batch, data_idxs, data_records)
                for idx, lbl in zip(new_in_batch, labels):
                    labeled[idx] = lbl

            for i in batch:
                all_obs.append(labeled[i])

            obs = np.array(all_obs)

            if len(obs) >= region_size:
                return np.mean(obs) >= self.target

            if np.mean(obs) < self.target:
                conf = test_if_true_mean_is_below_m(
                    obs, self.target, alpha=delta_val,
                    without_replacement=True, N=region_size,
                    fixed_sample_size=False
                )
                if not conf:
                    return False
            else:
                conf = test_if_true_mean_is_above_m(
                    obs, self.target, alpha=delta_val,
                    without_replacement=True, N=region_size,
                    fixed_sample_size=False
                )
                if not conf:
                    return True

        return np.mean(all_obs) >= self.target

    def _bargain_precision_search(self, t_R, data_idxs, data_records, n, labeled, delta_prec):
        """
        BARGAIN-style threshold search on [t_R, n) to find t_P where
        precision >= target in [t_P, n). Searches from right (small accept
        region) to left (large accept region).
        """
        region_size = n - t_R
        if region_size == 0:
            return n

        thresh_step = max(region_size // self.M, 1)
        sample_step = 10

        # Reversed region: index 0 = sorted index n-1 (highest proxy score)
        sampler = WoR_Sampler(region_size)
        best_thresh = -1

        for curr_thresh in range(thresh_step - 1, region_size, thresh_step):
            sampled_labels = []
            is_above = False

            while True:
                result = sampler.sample(curr_thresh, sample_step)
                if not isinstance(result, tuple):
                    if len(sampled_labels) > 0:
                        is_above = np.mean(sampled_labels) >= self.target
                    break

                sampled_rev, budget_used, sampled_all = result
                sorted_indices = (n - 1 - sampled_rev).astype(int)

                new_to_label = [si for si in sorted_indices if si not in labeled]
                if new_to_label:
                    labels = self._get_labels(new_to_label, data_idxs, data_records)
                    for idx, lbl in zip(new_to_label, labels):
                        labeled[idx] = lbl

                for si in sorted_indices:
                    sampled_labels.append(labeled[si])

                obs = np.array(sampled_labels)
                N = curr_thresh + 1

                if sampled_all:
                    is_above = np.mean(obs) >= self.target
                    break

                if np.mean(obs) < self.target:
                    conf = test_if_true_mean_is_below_m(
                        obs, self.target, alpha=delta_prec,
                        without_replacement=True, N=N,
                        fixed_sample_size=False
                    )
                    if not conf:
                        is_above = False
                        break
                else:
                    conf = test_if_true_mean_is_above_m(
                        obs, self.target, alpha=delta_prec,
                        without_replacement=True, N=N,
                        fixed_sample_size=False
                    )
                    if not conf:
                        is_above = True
                        break

            if is_above:
                best_thresh = curr_thresh
            else:
                break

        if best_thresh < 0:
            return n
        return n - 1 - best_thresh

    def process(self, data_records: List[str]) -> List[int]:
        '''
        Returns a set of data indexes estimated to be positive. It guarantees the set has
        precision >= target AND recall >= target with probability >= 1 - delta.

        Args:
            data_records: String array containing data records to be processed.

        Returns:
            List[int]: A list containing indexes of records in `data_records` estimated to be positive.
        '''
        self.proxy.reset()
        self.oracle.reset()

        data_records = np.array(data_records)
        data_idxs = np.arange(len(data_records))
        n = len(data_records)

        if self.verbose:
            print("Getting Proxy output and Scores")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(
            data_idxs, data_records
        )
        x_probs = proxy_preds * proxy_scores + (1 - proxy_preds) * (1 - proxy_scores)

        sort_idx = np.argsort(x_probs)
        x_probs = x_probs[sort_idx]
        data_idxs = data_idxs[sort_idx]
        data_records = data_records[sort_idx]

        labeled = {}

        # Phase 1: Find recall threshold t_R (using delta/2)
        if self.verbose:
            print("Finding Recall Threshold")
        t_R = self._find_recall_threshold(data_idxs, data_records, n, labeled)
        if self.verbose:
            print(f"Recall threshold at sorted index {t_R} ({t_R * 100 / n:.1f}% rejected)")

        # Phase 2: Find precision threshold t_P (using delta/2, split delta/4 + delta/4)
        if self.verbose:
            print("Finding Precision Threshold (sliding window)")
        candidate_t_P = self._sliding_window_precision(
            t_R, data_idxs, data_records, n, labeled
        )

        if candidate_t_P >= n:
            if self.verbose:
                print("Sliding window found no candidate, using BARGAIN fallback")
            t_P = self._bargain_precision_search(
                t_R, data_idxs, data_records, n, labeled, self.delta / 2
            )
        else:
            if self.verbose:
                print(f"Candidate precision threshold at sorted index {candidate_t_P}, validating...")
            is_valid = self._validate_precision(
                candidate_t_P, data_idxs, data_records, n, labeled, self.delta / 4
            )
            if is_valid:
                t_P = candidate_t_P
                if self.verbose:
                    print("Validation passed")
            else:
                if self.verbose:
                    print("Validation failed, using BARGAIN fallback")
                t_P = self._bargain_precision_search(
                    t_R, data_idxs, data_records, n, labeled, self.delta / 4
                )

        if self.verbose:
            print(f"Precision threshold at sorted index {t_P}")
            if n > 0:
                print(
                    f"  {t_R * 100 / n:.1f}% rejected, "
                    f"{(t_P - t_R) * 100 / n:.1f}% oracle, "
                    f"{(n - t_P) * 100 / n:.1f}% accepted by proxy"
                )

        # Phase 3: Final classification
        positive_sorted_indices = set(range(t_P, n))

        # Between t_R and t_P: label with oracle
        to_label_oracle = [i for i in range(t_R, t_P) if i not in labeled]
        if to_label_oracle:
            labels = self._get_labels(to_label_oracle, data_idxs, data_records)
            for idx, lbl in zip(to_label_oracle, labels):
                labeled[idx] = lbl

        for i in range(t_R, t_P):
            if labeled[i] == 1:
                positive_sorted_indices.add(i)

        # Below t_R: include already-labeled positives from sampling phases
        for i in range(0, t_R):
            if i in labeled and labeled[i] == 1:
                positive_sorted_indices.add(i)

        if len(positive_sorted_indices) == 0:
            return []

        sorted_indices_arr = np.array(sorted(positive_sorted_indices), dtype=int)
        result = data_idxs[sorted_indices_arr]
        return sorted(result.tolist())
