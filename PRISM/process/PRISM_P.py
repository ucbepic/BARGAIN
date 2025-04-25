import numpy as np
from typing import List

from PRISM.sampler.wor_sampler import WoR_Sampler
from PRISM.models.AbstractModels import Oracle, Proxy 
from PRISM.bounds.betting_bounds import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m

class PRISM_P():
    '''
    Class to process a dataset using a cheap proxy or an expensive oracle while guaranteeing a desired precision target
    '''
    def __init__(
                self,
                proxy: Proxy,
                oracle: Oracle, 
                delta:float=0.1,
                target:float=0.9,
                budget:int=400,
                M:int= 20,
                eta:int = 0,
                seed:int=0
            ) -> None:
        '''
        Args: 
            proxy: Proxy model to use 
            oracle: Oracle model to use 
            delta: Probability of failure, float between 0 and 1
            target: Desired precision target, float between 0 and 1
            bugdet: Maximum number of records that can be processed by the oracle
            M: Number of different thresholds to be considered by algorithm
            eta: Tolerance parameter
            seed: Random seed

        '''
        self.delta = delta
        self.target = target
        self.budget = budget
        self.eta = eta+1


        self.proxy = proxy
        self.oracle = oracle

        self.M = M
        if seed is not None:
            np.random.seed(seed)


    def __sample_till_confident(self, budget, all_data_indexes, confidence, target, total_sampled, curr_thresh, data_records):
        sample_step = 10
        sampled_label = np.array([])
        sampled_index = np.array([])
        
        while budget-total_sampled  > 0:
            no_sample = min(sample_step, budget-total_sampled)
            sampled_indexes, budget_used, sampled_all = self.sampler.sample(curr_thresh, no_sample)
            sampled_data_indexes = all_data_indexes[sampled_indexes]

            sampled_label = np.concatenate([sampled_label, self.oracle.get_pred(data_records[sampled_indexes], sampled_data_indexes)])
            sampled_index = np.concatenate([sampled_index, sampled_indexes])
            total_sampled += budget_used

            if sampled_all:
                return not np.mean(sampled_label)<target, sampled_index, sampled_label, total_sampled
                
            N = curr_thresh+1

            if np.mean(sampled_label)<target:
                conf_has_target = test_if_true_mean_is_below_m(np.array(sampled_label), target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False)
            else:
                conf_has_target = test_if_true_mean_is_above_m(np.array(sampled_label), target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False)

            if np.mean(sampled_label)<target:
                is_below_target = True
            else:
                is_below_target = False
            if not conf_has_target:
                return not is_below_target, sampled_index, sampled_label, total_sampled
            
        return False, sampled_index, sampled_label, total_sampled


    def process(self, data_records:List[str]) -> List[int]:
        '''
        Returns a set of data indexes estimated to be positive. It guarantees the set has precision at least equal to `target` with probability 1-`delta`

        Args:
            data_records: String array containing data records to be processed. 

        Returns:
            List[int]: A list containing indexes of records in `data_records` estimated to be positive.

        '''
        self.proxy.reset()
        self.oracle.reset()

        data_idxs = np.arange(len(data_records))
        data_records = np.array(data_records)
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(data_idxs, data_records)
        x_probs = proxy_preds*proxy_scores+(1-proxy_preds)*(1-proxy_scores)

        self.sampler = WoR_Sampler(len(data_idxs))
        thresh_step = max(len(data_idxs)//self.M, 1)

        sort_indx = np.argsort(x_probs)[::-1]
        x_probs = x_probs[sort_indx]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        budget = self.budget

        sample_labels =[]
        sample_indexes = []
        total_sampled = 0

        best_thresh = 0
        tries_used = 0
        for curr_thresh in range(thresh_step-1, len(data_idxs), thresh_step):
            is_confident_above_target, sampled_index, sampled_label, total_sampled = self.__sample_till_confident(budget, data_idxs, self.delta/self.eta, self.target, total_sampled,  curr_thresh, data_records)

            sample_labels = np.concatenate([sample_labels,sampled_label])
            sample_indexes = np.concatenate([sample_indexes,sampled_index])

            if budget==total_sampled:
                break
            if not is_confident_above_target:
                tries_used += 1
                if tries_used>=self.eta:
                    break
            else:
                best_thresh = curr_thresh

        if budget-total_sampled > 0 and best_thresh<len(data_idxs)-1:
            more_samples = []
            curr_to_label = best_thresh+1
            while budget-total_sampled > 0 and curr_to_label < len(data_idxs):
                more_samples.append(curr_to_label)
                if curr_to_label not in sample_indexes:
                    total_sampled += 1
                curr_to_label += 1
            more_samples = np.array(more_samples)
            sample_indexes = np.concatenate([sample_indexes, more_samples])
            total_sampled += len(more_samples)


        set_ids = data_idxs[:best_thresh]
        sample_indexes = np.unique(np.array(sample_indexes).astype(int))
        all_sample_indexs = data_idxs[sample_indexes]
        all_sample_labels = self.oracle.get_pred(all_sample_indexs)
        samp_inds = data_idxs[sample_indexes[all_sample_labels==1]]
        all_inds = np.unique( np.concatenate([set_ids, samp_inds]))
        return all_inds.tolist()


