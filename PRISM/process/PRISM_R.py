import numpy as np
from typing import List

from PRISM.sampler.wor_sampler import WoR_Sampler
from PRISM.models.AbstractModels import Oracle, Proxy 
from PRISM.bounds.betting_bounds import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m

class PRISM_R():
    '''
    Class to process a dataset using a cheap proxy or an expensive oracle while guaranteeing a desired recall target
    '''
    def __init__(
        self,
        proxy: Proxy,
        oracle: Oracle,
        delta:float=0.1,
        target:float=0.9,
        budget:int=400,
        beta:float=0,
        r:float=150,
        seed:int=0
    ):
        '''
        Args: 
            proxy: Proxy model to use 
            oracle: Oracle model to use 
            delta: Probability of failure, float between 0 and 1
            target: Desired recall target, float between 0 and 1
            budget: Maximum number of records that can be processed by the oracle
            beta: minimum density parameter. **WARNING** setting beta>0 changes the statistical guarantees
            r: resolution parameter. Only used if beta>0
            seed: Random seed

        '''
        self.delta = delta
        self.target = target
        self.budget = budget

        self.proxy = proxy
        self.oracle = oracle

        self.beta = beta
        self.r = r

        if seed is not None:
            np.random.seed(seed)

    def __sample_till_can_exclude(self, begin_n, delta, budget, all_data_indexes, total_sampled, data_records):
        sample_step = 10
        sampled_label = np.array([])
        sampled_index = np.array([])
        
        begin_thresh = begin_n
        end_thresh = begin_n+self.r
        while budget-total_sampled  > 0:
            no_sample = min(sample_step, budget-total_sampled)
            sampled_indexes, budget_used, sampled_all = self.sampler.sample_high_low(begin_thresh, end_thresh, no_sample)

            if len(sampled_indexes)==0:
                assert len(sampled_label)>0
                return np.mean(sampled_label)<self.beta, sampled_index, sampled_label, total_sampled

            sampled_data_indexes = all_data_indexes[sampled_indexes]
            sampled_label = np.concatenate([sampled_label, self.oracle.get_pred(data_records[sampled_indexes], sampled_data_indexes)])
            sampled_index = np.concatenate([sampled_index, sampled_indexes])
            total_sampled += budget_used

            if np.mean(sampled_label)>2*self.beta or sampled_all:
                return np.mean(sampled_label)<self.beta, sampled_index, sampled_label, total_sampled
                

            conf_has_tail_prob = test_if_true_mean_is_below_m(np.array(sampled_label), self.beta, alpha=delta, without_replacement=True, N=self.r, fixed_sample_size=False)

            if not conf_has_tail_prob:
                return True, sampled_index, sampled_label, total_sampled
    
        return False, sampled_index, sampled_label, total_sampled   

    def __find_sample_region_exp_search(self, data_idxs, data_records, budget, est_budget):
        est_delta = self.delta/2
        exp_search_delta = (self.delta-est_delta)/2
        binary_search_delta = (self.delta-est_delta-exp_search_delta)/2
        
        sample_labels =[]
        sample_indexes = []
        total_sampled = 0

        begin_n = len(data_idxs)-2*self.r
        while begin_n>=len(data_idxs)//2 and begin_n>=0:
            exp_search_delta = exp_search_delta/2
            is_tail, curr_sampled_index, curr_sampled_label, total_sampled  = self.__sample_till_can_exclude(begin_n, exp_search_delta, budget, data_idxs, total_sampled, data_records)

            sample_labels = np.concatenate([sample_labels,curr_sampled_label])
            sample_indexes = np.concatenate([sample_indexes,curr_sampled_index])


            if is_tail:
                curr_budget_left = budget-total_sampled
                if curr_budget_left + est_budget>= (len(data_idxs)-begin_n)/2:
                    return begin_n, binary_search_delta,curr_budget_left
                begin_n, delta_left, budget_left = self.__find_sample_region_binary_search(begin_n, data_idxs, curr_budget_left, binary_search_delta, data_records)
                return  begin_n, delta_left, budget_left

            begin_n = len(data_idxs)- 2*(len(data_idxs)-begin_n)

        return 0, exp_search_delta+binary_search_delta, budget-total_sampled



    def __find_sample_region_binary_search(self, begin_range, data_idxs, budget, bin_search_delta, data_records):
        
        sample_labels =[]
        sample_indexes = []
        total_sampled = 0

        last_valid = begin_range
        begin_n = (len(data_idxs)+begin_range)//2
        while True:
            is_tail, curr_sampled_index, curr_sampled_label, total_sampled  = self.__sample_till_can_exclude(begin_n, bin_search_delta, budget, data_idxs, total_sampled, data_records)

            sample_labels = np.concatenate([sample_labels,curr_sampled_label])
            sample_indexes = np.concatenate([sample_indexes,curr_sampled_index])


            if not is_tail:
                return last_valid, 0,budget-total_sampled

            last_valid = begin_n
            begin_n = (len(data_idxs)+begin_n)//2

    def __find_max_positive(self, sample_indx, delta, threshs_to_try):
        best_t  = None
        for t in threshs_to_try:
            samples_at_thresh = sample_indx>=t
            if np.mean(samples_at_thresh)<self.target:
                conf_has_target = True
            else:
                conf_has_target = test_if_true_mean_is_above_m(np.array(samples_at_thresh), self.target, alpha=delta)
            if conf_has_target:
                return best_t
            else:
                best_t = t
        return None

    def __process_uniform(self, data_records) -> np.ndarray:
        data_idxs = np.arange(len(data_records))
        data_records = np.array(data_records)
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(data_idxs, data_records)
        x_probs = proxy_preds*proxy_scores+(1-proxy_preds)*(1-proxy_scores)

        sort_indx = np.argsort(x_probs)
        x_probs = x_probs[sort_indx]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        sampled_indexes = np.random.choice(len(data_idxs), self.budget, replace=True)
        sample_data_indx = data_idxs[sampled_indexes]
        sampled_label =  self.oracle.get_pred(data_records[sampled_indexes], sample_data_indx)
        pos_sampled_indexes = sampled_indexes[sampled_label==1]
        threshs_to_try = np.sort(pos_sampled_indexes)

        best_t  = None
        for t in threshs_to_try:
            samples_at_thresh = pos_sampled_indexes>=t
            if np.mean(samples_at_thresh)<self.target:
                conf_has_target = True
            else:
                conf_has_target = test_if_true_mean_is_above_m(np.array(samples_at_thresh), self.target, alpha=self.delta)
            if conf_has_target:
                break
            else:
                best_t = t
        if best_t is None:
            best_t = 0

        indexes_assumed_positive = np.arange(best_t, len(data_idxs)).astype(int)
        set_ids = data_idxs[indexes_assumed_positive]
        samp_inds = data_idxs[sampled_indexes[sampled_label==1]]
        all_inds = np.unique( np.concatenate([set_ids, samp_inds]))
        return all_inds

    def process(self, data_records:List[str]) -> List[int]:
        '''
        Returns a set of data indexes estimated to be positive. It guarantees the set has recall at least equal to `target` with probability 1-`delta`

        Args:
            data_records: String array containing data records to be processed. 

        Returns:
            List[int]: A list containing indexes of records in `data_records` estimated to be positive.

        '''
        self.proxy.reset()
        self.oracle.reset()

        if self.beta == 0:
            return self.__process_uniform(data_records).tolist()

        data_idxs = np.arange(len(data_records))
        data_records = np.array(data_records)
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(data_idxs, data_records)
        x_probs = proxy_preds*proxy_scores+(1-proxy_preds)*(1-proxy_scores)
        

        self.sampler = WoR_Sampler(len(data_idxs))

        sort_indx = np.argsort(x_probs)
        x_probs = x_probs[sort_indx]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        region_budget = self.budget//2
        est_budget = self.budget-region_budget
        est_delta = self.delta/2
        delta_left = self.delta-est_delta

        begin_n, delta_left, budget_left = self.__find_sample_region_exp_search( data_idxs, data_records, region_budget, est_budget)


        sampled_indexes = np.random.choice(np.arange(begin_n, len(data_idxs)), est_budget+budget_left)
        sample_data_idxs = data_idxs[sampled_indexes]
        sampled_labels = self.oracle.get_pred(sample_data_idxs)

        est_delta = est_delta+delta_left
        curr_positives = sampled_indexes[sampled_labels  == True]
        threshs_to_try = np.sort(curr_positives)
        best_recall = self.__find_max_positive(curr_positives, est_delta, threshs_to_try)
        if best_recall is None:
            best_recall = begin_n

        pos_samples = data_idxs[sampled_indexes[sampled_labels  == True]]
        thresh_samples = data_idxs[best_recall:]

        all_inds = np.unique(
                np.concatenate([pos_samples, thresh_samples])
        )
        assert self.oracle.get_number_preds() <= self.budget
        return all_inds.tolist()
