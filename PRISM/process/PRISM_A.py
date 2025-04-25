import numpy as np
from typing import List, Union, Tuple

from PRISM.sampler.wor_sampler import WoR_Sampler
from PRISM.models.AbstractModels import Oracle, Proxy 
from PRISM.bounds.betting_bounds import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m


class PRISM_A():
    '''
    Class to process a dataset using a cheap proxy or an expensive oracle while guaranteeing the output is validatd by the oracle with a desired accuracy target
    '''
    def __init__(
                self,
                proxy: Proxy,
                oracle: Oracle,
                target:float=0.9,
                delta:float=0.1,
                M:int= 20,
                verbose:bool=True,
                seed:int=0
            ) -> None:
        '''
        Args: 
            proxy: Proxy model to use 
            oracle: Oracle model to use 
            target: Desired precision target, float between 0 and 1
            delta: Probability of failure, float between 0 and 1
            M: Number of different thresholds to be considered by algorithm
            verbose: output progress details or not
            seed: Random seed

        '''
        self.delta = delta
        self.target = target

        self.proxy = proxy
        self.oracle = oracle

        self.M = M
        if seed is not None:
            np.random.seed(seed)
        self.verbose = verbose
  
    def __check_worth_trying(self, sample_indx, sample_is_correct, t, target):
        if len(sample_indx) < 50:
            return True
        mask_at_t = sample_indx<=t
        samples_at_thresh = sample_is_correct[mask_at_t]
        if np.mean(samples_at_thresh)-np.std(samples_at_thresh)<target:
            return False
        return True

    def __sample_till_confident_above_target(self, all_data_indexes, all_preds, confidence, target, total_sampled, curr_thresh, data_records):
        sample_step = 10
        sampled_is_correct = np.array([])
        sampled_preds = np.array([])
        sampled_index = np.array([]).astype(int)
        
        while self.__check_worth_trying(sampled_index, sampled_is_correct, curr_thresh, target):
            sampled_indexes, budget_used, sampled_all = self.sampler.sample(curr_thresh, sample_step)

            sampled_data_indexes = all_data_indexes[sampled_indexes]
            proxy_preds = all_preds[sampled_indexes]
            sampled_is_correct = np.concatenate([sampled_is_correct, self.oracle.is_answer_correct(sampled_data_indexes, data_records[sampled_indexes], proxy_preds)])
            sampled_index = np.concatenate([sampled_index, sampled_indexes])
            sampled_preds = np.concatenate([sampled_preds, proxy_preds])
            total_sampled += budget_used

            if sampled_all:
                return not np.mean(sampled_is_correct)<target, sampled_index, total_sampled
                
            samples_at_thresh = sampled_is_correct[sampled_index<=curr_thresh]
            N = curr_thresh+1
            if np.mean(samples_at_thresh)<target:
                conf_has_target = test_if_true_mean_is_below_m(samples_at_thresh, target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False)
            else:
                conf_has_target = test_if_true_mean_is_above_m(samples_at_thresh, target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False)

            if np.mean(samples_at_thresh)<target:
                is_below_target = True
            else:
                is_below_target = False
            if not conf_has_target:
                return not is_below_target, sampled_index, total_sampled
            
        return False, sampled_index, total_sampled

    def process(self, data_records:List[str], return_oracle_usage:bool= False) -> Union[List[str], Tuple[List[str], List[bool]]]:
        '''
        Returns the computed output for all data records. It guarantees the output matches what the `oracle` would've provided on at least `target` fraction of the records with probability 1-`delta` but minimizes number of `oracle` usags
        Args:
            data_records: String array containing data records to be processed. 
            return_oracle_usage: If `True`, the function additionally outputs whether a record was processed by oracle or not

        Returns:
            Union[List[str], Tuple[List[str], List[bool]]]: 
                - If `return_oracle_usage` is False, returns a list of processed output strings:
                    - List[str]: The computed outputs for the input `data_records` in the same order as `data_records`
                - If `return_oracle_usage` is True, returns a tuple:
                    - List[str]: The computed outputs for the input `data_records` in the same order as `data_records`
                    - List[bool]: A list of booleans where each element indicates whether the oracle was used for that record.

        '''
        self.proxy.reset()
        self.oracle.reset()

        data_records = np.array(data_records)
        data_idxs = np.arange(len(data_records))
        self.sampler = WoR_Sampler(len(data_idxs))
        thresh_step = max(len(data_idxs)//self.M, 1)

        if self.verbose:
            print("Getting Proxy output and Scores")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(data_idxs, data_records[data_idxs])

        sort_indx = np.argsort(proxy_scores)[::-1]
        proxy_preds = proxy_preds[sort_indx]
        proxy_scores = proxy_scores[sort_indx]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]


        sample_indexes = []
        total_sampled = 0

        best_thresh = 0
        if self.verbose:
            print("Determining Cascade Threshold")
        for curr_thresh in range(thresh_step-1, len(data_idxs), thresh_step):
            if curr_thresh == len(data_idxs)-1:
                new_target = self.target
            else:
                n_from_proxy = curr_thresh+1
                n_from_oracle = len(data_idxs)-n_from_proxy
                new_target = (self.target*(n_from_oracle+n_from_proxy)-n_from_oracle)/n_from_proxy
                if new_target <= 0:
                    continue

            is_confident_above_target, sampled_index, total_sampled = self.__sample_till_confident_above_target(data_idxs, proxy_preds, self.delta, new_target, total_sampled,  curr_thresh, data_records)

            sample_indexes = np.concatenate([sample_indexes,sampled_index])

            if not is_confident_above_target:
                break
            best_thresh = curr_thresh
        proxy_indxs = np.setdiff1d(data_idxs[:best_thresh], data_idxs[np.array(sample_indexes).astype(int)])

        if self.verbose:
            print(f"Found Threshold, {len(proxy_indxs)*100/len(data_idxs):.1f}% of Data is Processed with Proxy")

        oracle_indexes = np.setdiff1d(data_idxs, proxy_indxs)
        if self.verbose:
            print(f"Processing with Oracle")
        oracle_outputs = self.oracle.get_pred(data_records[oracle_indexes], oracle_indexes)

        if self.verbose:
            print(f"Processing with Proxy")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(proxy_indxs, data_records[proxy_indxs])
        indexes_data_indx = np.concatenate([oracle_indexes, proxy_indxs])
        output = np.concatenate([oracle_outputs, proxy_preds])

        order = np.argsort(indexes_data_indx)
        output  = output[order]

        if return_oracle_usage:
            used_oracle = np.array([True]*len(oracle_indexes)+[False]*len(proxy_indxs))
            used_oracle  = used_oracle[order]
            return output.tolist(), used_oracle.tolist()

        return output.tolist()

