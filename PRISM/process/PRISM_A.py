import numpy as np
import pandas as pd

from PRISM.sampler.wor_sampler import WoR_Sampler
from PRISM.models.AbstractModels import Oracle, Proxy 
from PRISM.bounds.betting_bounds import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m


class PRISM_A():
    '''
    Class to process a dataset using a cheap proxy or an expensive oracle while guaranteeing the output is validatd by the oracle with a desired accuracy target
    '''
    def __init__(
                self,
                data_indxes: np.ndarray,
                proxy: Proxy,
                oracle: Oracle,
                delta:float=0.1,
                target:float=0.9,
                M:int= 20,
                verbose:bool=True,
                seed:int=0
            ):
        '''
        Args: 
            data_indxes: Identifies for each data record to be processed. The identifiers are passed to `proxy` or `oracle` to process a record. Outputs also use these identifiers to refer to a data record. 
            proxy: Proxy model to use 
            oracle: Oracle model to use 
            delta: Probability of failure, float between 0 and 1
            target: Desired precision target, float between 0 and 1
            M: Number of different thresholds to be considered by algorithm
            verbose: output progress details or not
            seed: Random seed

        '''
        self.delta = delta
        self.target = target
        self.data_indexs = data_indxes

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

    def __sample_till_confident_above_mean(self, all_data_indexes, all_preds, confidence, target, total_sampled, curr_thresh):
        sample_step = 10
        sampled_is_correct = np.array([])
        sampled_preds = np.array([])
        sampled_index = np.array([]).astype(int)
        
        while self.__check_worth_trying(sampled_index, sampled_is_correct, curr_thresh, target):
            sampled_indexes, budget_used, sampled_all = self.sampler.sample(curr_thresh, sample_step)

            sampled_data_indexes = all_data_indexes[sampled_indexes]
            proxy_preds = all_preds[sampled_indexes]
            sampled_is_correct = np.concatenate([sampled_is_correct, self.oracle.is_answer_correct(sampled_data_indexes, proxy_preds)])
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

    def process(self) -> pd.DataFrame:
        '''
        Returns the computed output for all data records. It guarantees the output matches what the `oracle` would've provided on at least `target` fraction of the records with probbility 1-`delta` but minimizes number of `oracle` usags

        Returns:
            pd.DataFrame: A dataframe indexed by the `data_indxes`, with columns `output` and `used_oracle` where each row corresponds to a data record, `output` column shows the computed output for the record and `used_oracle` denotes whether the oracle was used to compute the output for that record or not

        '''
        data_idxs = self.data_indexs
        self.sampler = WoR_Sampler(len(data_idxs))
        thresh_step = max(len(data_idxs)//self.M, 1)

        if self.verbose:
            print("Getting Proxy output and Scores")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(data_idxs)

        sort_indx = np.argsort(proxy_scores)[::-1]
        proxy_preds = proxy_preds[sort_indx]
        proxy_scores = proxy_scores[sort_indx]
        data_idxs = data_idxs[sort_indx]


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

            is_confident_above_target, sampled_index, total_sampled = self.__sample_till_confident_above_mean(data_idxs, proxy_preds, self.delta, new_target, total_sampled,  curr_thresh)

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
        oracle_outputs = self.oracle.get_pred(oracle_indexes)
        df_oracle = pd.DataFrame.from_dict({'data_indx':oracle_indexes, 'output':oracle_outputs, 'used_oracle':[True]*len(oracle_indexes)})

        if self.verbose:
            print(f"Processing with Proxy")
        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(proxy_indxs)
        df_proxy = pd.DataFrame.from_dict({'data_indx':proxy_indxs, 'output':proxy_preds, 'used_oracle':[False]*len(proxy_indxs)})

        res_df = pd.concat([df_oracle, df_proxy]).set_index('data_indx')
        return res_df

